## 2D convolution

For image processing and computer vision, input data is typically represented as 2D arrays, with pixels in an x–y space. Image convolutions are therefore **2D convolutions**, as illustrated in **Fig. 7.4 (Pic 1)**.

In a 2D convolution the filter `f` is also a 2D array. Its x and y dimensions determine the range of neighbours to be included in the weighted sum calculation. If we assume that the dimension of the filter is $(2r_x + 1)$ in the x dimension and $(2r_y + 1)$ in the y dimension, the calculation of each output element `P` can be expressed as:

![Image 3](https://github.com/user-attachments/assets/b408a7a7-9976-44b6-8f0e-6b180a79649b)


$$
P[y, x] =
\sum_{j=-r_y}^{r_y}
\sum_{i=-r_x}^{r_x}
N[y + j, x + i] \cdot f[j + r_y, i + r_x]
$$

**(Honestly, don'f focus too much on the equation you already know what it means conceptually)

In Fig. 7.4 we use a $5 \times 5$ filter for simplicity; that is, $r_y = 2$ and $r_x = 2$. In general, the filter does not have to be, but is typically, a square array.

To generate an output element, we take the subarray whose centre is at the corresponding location in the input array `N`. We then perform pairwise multiplication between elements of the filter array and those of the image array. For our example the result is shown as the $5 \times 5$ product array below `N` and `P` in Fig. 7.4. The value of the output element is the sum of all elements of the product array.


## Worked example: computing $P_{2,2}$

The example in Fig. 7.4 shows the calculation of $P_{2,2}$.

For brevity, we use $N_{y,x}$ to denote `N[y][x]` in a C-style array. Since `N` and `P` are most likely dynamically allocated arrays, our actual code uses linearised indices, but the maths is easier to read in 2D form.

The calculation is:

$$
\begin{aligned}
P_{2,2} =\;&
N_{0,0}M_{0,0} + N_{0,1}M_{0,1} + N_{0,2}M_{0,2} + N_{0,3}M_{0,3} + N_{0,4}M_{0,4} \\
&+ N_{1,0}M_{1,0} + N_{1,1}M_{1,1} + N_{1,2}M_{1,2} + N_{1,3}M_{1,3} + N_{1,4}M_{1,4} \\
&+ N_{2,0}M_{2,0} + N_{2,1}M_{2,1} + N_{2,2}M_{2,2} + N_{2,3}M_{2,3} + N_{2,4}M_{2,4} \\
&+ N_{3,0}M_{3,0} + N_{3,1}M_{3,1} + N_{3,2}M_{3,2} + N_{3,3}M_{3,3} + N_{3,4}M_{3,4} \\
&+ N_{4,0}M_{4,0} + N_{4,1}M_{4,1} + N_{4,2}M_{4,2} + N_{4,3}M_{4,3} + N_{4,4}M_{4,4}
\end{aligned}
$$

In the numeric example shown in the figure, this corresponds to summing all elements of the $5 \times 5$ product array.


## Boundary conditions in 2D convolution

Like 1D convolution, 2D convolution must also deal with boundary conditions. With boundaries in both the x and y dimensions, the calculation of an output element may involve boundary conditions along a horizontal boundary, a vertical boundary, or both.

Fig. 7.5 illustrates the calculation of a `P` element that involves both boundaries.

From Fig. 7.5, the calculation of $P_{1,0}$ involves two missing columns and one missing row in the subarray of `N`. As in 1D convolution, different applications assume different default values for these missing `N` elements. In this example we assume the default value is `0`.

These boundary conditions also affect the efficiency of tiling. We will come back to this point soon.

![Image 4](https://github.com/user-attachments/assets/4db6ae54-c43f-4cd3-8437-cb953681e3de)

