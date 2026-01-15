# Online softmax: streaming stability for long rows

## Why online softmax exists

Safe softmax is numerically stable, but it relies on being able to compute a **row wise maximum** and a **row wise normalisation term** in one go:

$$
y_i = \frac{e^{x_i - m}}{\sum_{j=0}^{N-1} e^{x_j - m}}
\quad \text{where} \quad
m = \max_{j} x_j
$$

If the entire softmax row fits inside a single CUDA block, this is easy to implement using two block reductions:

- one block max reduction to compute $$m$$
- one block sum reduction to compute $$\sum e^{x_j - m}$$

However, in practice, especially in attention, the row length `N` (for example `KV_LEN`) can be **much larger than 1024**, which is the maximum number of threads per block. This means:

- one block cannot cover the entire row
- the row must be processed in **chunks (tiles)**

Online softmax exists to solve exactly this problem: computing a **numerically stable softmax while streaming over chunks of a long row**, without needing extra global reductions or multiple kernels.


## The key idea behind online softmax

Online softmax maintains **two running quantities** while scanning the row:

- a running maximum value, denoted $$m$$
- a running normalisation term, denoted $$l$$

After processing some prefix of the row, these satisfy:

$$
m = \max(\text{values seen so far})
$$

$$
l = \sum_{\text{values seen so far}} e^{x - m}
$$

The crucial difficulty is that when new values are processed, the maximum may increase. When the maximum increases, **all previously accumulated exponentials must be rescaled** so they remain correct relative to the new maximum.

That rescaling is the core idea behind online softmax.


## The online softmax update equations (streaming form)

Assume we are processing the row element by element.

Let:

- $$m_{i-1}$$ be the running max after processing elements $$0 \ldots i-1$$
- $$l_{i-1}$$ be the running normaliser after processing elements $$0 \ldots i-1$$

Now we process a new element $$x_i$$.

### Step 1: update the max

$$
m_i = \max(m_{i-1}, x_i)
$$

### Step 2: update the normaliser

$$
l_i = l_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}
$$

This equation does two things:

- rescales the old normaliser if the max increased
- adds the new element’s contribution relative to the new max

After processing all elements, the final softmax is:

$$
y_i = \frac{e^{x_i - m}}{l}
$$

This is mathematically equivalent to safe softmax, but it is **streamable**.


## From streaming to parallel: merging summaries

The equations above describe **sequential streaming**: one element at a time.

On the GPU, we want **parallelism**, so instead of processing one element at a time, we process **groups of elements** and then merge their summaries.

To do this, we generalise the running state into a **summary struct**:

- $$m$$: the maximum of a group
- $$d$$: the normalisation term for that group

In code, this appears as:

```cpp
struct MD {
  float m; // max
  float d; // normaliser
};
```

Each `MD` represents the softmax summary of some subset of elements.


## Merging two summaries: the core equation

Suppose we have two summaries:

- **Summary A:** $$(m_a, d_a)$$  
- **Summary B:** $$(m_b, d_b)$$

Each summary represents the contribution of a subset of elements:

$$
d_a = \sum_{x \in A} e^{x - m_a}
\quad\quad
d_b = \sum_{x \in B} e^{x - m_b}
$$

We want to merge these into a single summary representing the union $$A \cup B$$.

### Step 1: compute the combined max

$$
m = \max(m_a, m_b)
$$

### Step 2: rescale and combine normalisers

Both normalisers must be expressed relative to the new max $$m$$. This gives:

$$
d = d_a \cdot e^{m_a - m} + d_b \cdot e^{m_b - m}
$$

This equation is the **direct parallel analogue** of the streaming update equation used in online softmax.


## How this appears in the CUDA code

In the warp reduction, each lane holds one `MD` value called `value`, and it fetches another one called `other` using a shuffle instruction.

The code then determines which of the two summaries has the larger maximum and merges them using the equation above.

```cpp
bool value_bigger = (value.m > other.m);
MD bigger_m = value_bigger ? value : other;
MD smaller_m = value_bigger ? other : value;
```

Then it applies the merge equation:

```cpp
value.d = smaller_m.d * expf(smaller_m.m - bigger_m.m) + bigger_m.d;
value.m = bigger_m.m
```

This matches the math exactly:

- `bigger_m.m` is the combined maximum $$m$$  
- `bigger_m.d` is already scaled correctly relative to $$m$$  
- `smaller_m.d * exp(smaller_m.m - bigger_m.m)` rescales the smaller summary so it is expressed relative to $$m$$  

This is **online softmax expressed as a parallel reduction operator**.


## Why this works as a warp or block reduction

The merge operation has three key properties:

1. **Associativity**  
   Summaries can be merged in any order without changing the final result.

2. **Numerical stability**  
   Exponentials are always taken relative to a local maximum, preventing overflow.

3. **Parallelisability**  
   Summaries can be reduced using warp shuffles and shared memory just like sums or maxima.

Because of these properties, the same logic works:

- within a warp  
- across warps inside a block  
- across tiles in FlashAttention style kernels  


## Connecting back to the streaming equations

The streaming update equation:

$$
l_i = l_{i-1} \cdot e^{m_{i-1} - m_i} + e^{x_i - m_i}
$$

is simply a **special case** of the merge equation where:

- one summary contains many elements  
- the other summary contains exactly one element  

In code, a single element is represented as:

```cpp
val.m = x_i;
val.d = 1.0f;
```

because:

$$
e^{x_i - x_i} = 1
$$


## GPU mapping for online per token softmax

Putting it all together:

- Each thread starts with an `MD` representing one element.
- Warp reductions merge these into per warp summaries.
- Block reductions merge warp summaries into a block summary.
- The final summary contains the correct running maximum $$m$$ and normaliser $$l$$ for the entire row or tile.
- Outputs are normalised using:

$$
y_i = \frac{e^{x_i - m}}{l}
$$

In FlashAttention, this process is **fused** with the computation of $$QK^T$$ and the multiplication with $$V$$, so softmax weights are never written to global memory. The softmax is applied on the fly while streaming through the data.


## Main takeaway

Online softmax is **not a different mathematical operation** from safe softmax. It is the **streaming and mergeable formulation** of the same computation.

The `MD` struct and the warp reduction logic are simply a parallel way of implementing the same recurrence:

- track a running maximum  
- track a correctly rescaled normaliser  
- merge summaries instead of individual elements  

This formulation is what makes long sequence attention and fused kernels like FlashAttention possible and efficient.
