# 1D Convolution on the GPU

Convolution is an array operation in which each output data element is a weighted sum of the corresponding input element and a collection of input elements that are centered on it. The weights that are used in the weighted sum calculation are defined by a filter array, commonly referred to as the convolution kernel. Since there is an unfortunate name conflict between the CUDA kernel functions and convolution kernels, we will refer to these filter arrays as convolution filters to avoid confusion.

Convolution can be performed on input data of different dimensionality: one-dimensional (1D) (e.g., audio), two-dimensional (2D) (e.g., photo), three-dimensional (3D) (e.g., video), and so on. In audio digital signal processing, the input 1D array elements are sampled signal volume over time. That is, the input data element xi is the ith sample of the audio signal volume. A convolution on 1D data, referred to as 1D convolution, is mathematically defined as a function that takes an input data array of n elements [x0, x1, …, xn−1] and a filter array of 2r + 1 elements [f0, f1, …, f2r] and returns an output data array y:


Since the size of the filter is an odd number (, the weighted sum calculation is symmetric around the element that is being calculated. That is, the weighted sum involves  input elements on each side of the position that is being calculated, which is the reason why  is referred to as the radius of the filter.

Fig. 7.1 shows a 1D convolution example in which a five-element (r = 2) convolution filter f is applied to a seven-element input array x. We will follow the C language convention by which x and y elements are indexed from 0 to 6 and f elements are indexed from 0 to 4. Since the filter radius is 2, each output element is calculated as the weighted sum of the corresponding input element, two elements on the left, and two elements on the right.

![Image](https://github.com/user-attachments/assets/4afb9077-710f-4317-8e47-6855f1365503)

For a concrete example, consider computing `y[2]` with a filter radius `r = 2`. That means `y[2]` uses the input values from two positions to the left up to two positions to the right:

- Left edge: `x[2 - 2] = x[0]`
- Right edge: `x[2 + 2] = x[4]`

So `y[2]` is a weighted sum of `x[0]` through `x[4]`.

In this example, assume:

- Input array:
`x = [8, 2, 5, 4, 1, 7, 3]`

- Filter weights:
`f = [1, 3, 5, 3, 1]`

To compute `y[2]`, you multiply each input value by its corresponding filter weight and then sum the products:

- `x[0] * f[0]`
- `x[1] * f[1]`
- `x[2] * f[2]`
- `x[3] * f[3]`
- `x[4] * f[4]`

You can also think of this as an inner product between a sliding window of `x` and the filter `f`.

More generally, `y[i]` is the inner product of:
- the subarray `x[i - r .. i + r]`
- with the filter `f[0 .. 2r]`

For example, `y[3]` is just the same pattern shifted one step to the right. It uses `x[1]` through `x[5]`:

- Left edge: `x[3 - 2] = x[1]`
- Right edge: `x[3 + 2] = x[5]`

So `y[3]` is the weighted sum of `x[1], x[2], x[3], x[4], x[5]` using the same filter weights.

![Image 1](https://github.com/user-attachments/assets/87aff2f7-5477-4467-99a4-0838248edbc4)

Because convolution uses neighbouring elements, we immediately run into **boundary conditions** near the start and end of the input array.

For example, when computing `y[1]` with radius `r = 2`, we would normally need the inputs:

- `x[1 - 2] = x[-1]`
- `x[1 - 1] = x[0]`
- `x[1]`
- `x[1 + 1] = x[2]`
- `x[1 + 2] = x[3]`

But `x[-1]` does not exist. So we cannot directly apply the “use 2 elements on each side” rule at the boundary.

A common way to handle this is to **pretend the missing elements exist and give them a default value**.  
In most applications, that default value is **0**. This is what Fig. 7.3 uses.

In audio processing, this makes intuitive sense: we can assume the signal is `0` before recording starts and after it ends.

So for `y[1]`, we treat the missing `x[-1]` as `0`, and then compute the weighted sum normally using the filter.

![Image 2](https://github.com/user-attachments/assets/b7223cd1-c706-4deb-8d62-0371a51cf6e5)

The input element that does not exist is shown as a dashed box in Fig. 7.3.

Once you see that for `y[1]`, it should be clear that `y[0]` is even “worse”: with radius `r = 2`, the computation for `y[0]` would require two elements to the left:

- `x[-2]`
- `x[-1]`

Both are missing, and in this example we treat both of them as `0`.

In the literature, these “missing but assumed” elements are often called **ghost cells**.

There is an important practical detail here: ghost cells do not only appear because of array boundaries. They also show up when we do **tiling** in parallel code. In that setting, each tile needs a small “halo” region of neighbouring values to compute its boundary outputs. Those halo values behave like ghost cells from the point of view of a tile. The size of that halo can strongly affect the effectiveness and efficiency of tiling.

Also, not all applications assume ghost cells are `0`. Other common boundary rules include:
- **clamp**: use the closest valid edge value (repeat `x[0]` on the left, repeat `x[n-1]` on the right)
