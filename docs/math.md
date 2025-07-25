## Patch Extraction

**Scenario:**

* **Input Vector $Y$:** A 1D input signal/vector.
* **Filter $W_1$:** A 1D filter (often called a kernel).

**Example:**

Let $Y = [y_1, y_2, y_3, y_4, y_5]$ be our input vector.
Let $W_1 = [w_1, w_2, w_3]$ be our 1D filter of support $f_1=3$.

The convolution operation, denoted by '$\ast$', for a 1D signal can be expressed as:

$$(Y \ast W_1)[i] = \sum_{k=1}^{f_1} Y[i-k+1] \cdot W_1[k]$$

Let's assume "same" padding for simplicity, where the output size is the same as the input size. Or, more simply, let's just show the valid convolutions without padding.

For a specific output element $F_1(Y)[i]$:

* **$F_1(Y)[1]$ (first valid output):**
    $(Y \ast W_1)[1] = Y[1] \cdot W_1[3] + Y[2] \cdot W_1[2] + Y[3] \cdot W_1[1]$ (assuming filter flipping for cross-correlation which is common in neural networks)
    Or more intuitively, without flipping (which is technically cross-correlation, often just called convolution in deep learning):
    $(Y \ast W_1)[1] = Y[1] \cdot W_1[1] + Y[2] \cdot W_1[2] + Y[3] \cdot W_1[3]$

Let's use the latter, which is often implemented as convolution in deep learning frameworks (cross-correlation).

Given:
$Y = [10, 20, 30, 40, 50]$
$W_1 = [0.1, 0.5, 0.1]$ (a simple blur filter example)

And let's say we have a bias $B_1 = 2$.

The operation $W_1 \ast Y$ with the "valid" padding (no padding, output smaller than input):

* **Output element 1 ($i=1$):**
    $(W_1 \ast Y)[1] = (Y[1] \cdot W_1[1]) + (Y[2] \cdot W_1[2]) + (Y[3] \cdot W_1[3])$
    $= (10 \cdot 0.1) + (20 \cdot 0.5) + (30 \cdot 0.1)$
    $= 1 + 10 + 3 = 14$

* **Output element 2 ($i=2$):**
    $(W_1 \ast Y)[2] = (Y[2] \cdot W_1[1]) + (Y[3] \cdot W_1[2]) + (Y[4] \cdot W_1[3])$
    $= (20 \cdot 0.1) + (30 \cdot 0.5) + (40 \cdot 0.1)$
    $= 2 + 15 + 4 = 21$

* **Output element 3 ($i=3$):**
    $(W_1 \ast Y)[3] = (Y[3] \cdot W_1[1]) + (Y[4] \cdot W_1[2]) + (Y[5] \cdot W_1[3])$
    $= (30 \cdot 0.1) + (40 \cdot 0.5) + (50 \cdot 0.1)$
    $= 3 + 20 + 5 = 28$

So, the result of $W_1 \ast Y$ would be approximately $[14, 21, 28]$.

Now, let's apply the full formula from the image: $F_1(\mathbf{Y}) = \max(0, W_1 \ast \mathbf{Y} + B_1)$

Using our calculated values and $B_1 = 2$:

* For output element 1:
    $F_1(Y)[1] = \max(0, (W_1 \ast Y)[1] + B_1) = \max(0, 14 + 2) = \max(0, 16) = 16$

* For output element 2:
    $F_1(Y)[2] = \max(0, (W_1 \ast Y)[2] + B_1) = \max(0, 21 + 2) = \max(0, 23) = 23$

* For output element 3:
    $F_1(Y)[3] = \max(0, (W_1 \ast Y)[3] + B_1) = \max(0, 28 + 2) = \max(0, 30) = 30$

Therefore, $F_1(Y) = [16, 23, 30]$ in this 1D example.

**Important Note for Image Convolution:**

In the context of image processing (2D convolution), $Y$ would be a 2D matrix (the image), and $W_1$ would be a 2D filter (kernel). The convolution operation involves sliding the 2D filter over the 2D image and performing element-wise multiplication and summation at each position. If there are multiple channels ($c$), the filter would also have $c$ channels, and the convolution would sum over all input channels. If $W_1$ corresponds to $n_1$ filters, then $n_1$ separate convolution operations would be performed, each yielding a 2D output feature map.


## Convolution Operation

Let:

* Input vector (signal): $x = [x_1, x_2, \dots, x_n]$
* Kernel/filter vector (weights): $w = [w_1, w_2, \dots, w_k]$

The **convolution operation** outputs a new vector $y$, defined as:

$$
y[i] = \sum_{j=1}^{k} w[j] \cdot x[i + j - 1]
$$

If $x$ has length $n$, and $w$ has length $k$, then:

* **Valid convolution** (no padding): $y$ has length $n - k + 1$
* **Same convolution** (with padding): $y$ has length $n$

---

### ⚠️ Convolution vs. Cross-Correlation

In many machine learning libraries (e.g., PyTorch, TensorFlow), the operation called “convolution” is actually **cross-correlation**, where the kernel is **not reversed**.

True convolution is:

$$
y[i] = \sum_{j=1}^{k} w[j] \cdot x[i - j + 1]
$$

Cross-correlation is:

$$
y[i] = \sum_{j=1}^{k} w[j] \cdot x[i + j - 1]
$$

---

### ✅ **Example**

Let’s say:

* $x = [1, 2, 3, 4, 5]$
* $w = [2, 1]$ (a filter of size 2)

Then:

**Valid convolution** gives:

$$
\begin{align*}
y[1] &= 2 \cdot 1 + 1 \cdot 2 = 4 \\
y[2] &= 2 \cdot 2 + 1 \cdot 3 = 7 \\
y[3] &= 2 \cdot 3 + 1 \cdot 4 = 10 \\
y[4] &= 2 \cdot 4 + 1 \cdot 5 = 13 \\
\end{align*}
$$

So result: $y = [4, 7, 10, 13]$

---

### 📌 In CNNs

When applied in Convolutional Neural Networks (CNNs), this operation:

* Slides the kernel over the input
* Computes dot products
* Produces a **feature map** for learning spatial or temporal patterns

## Support Meaning

### ✅ 1. **Kernel Size = Length**

In 1D convolution:

* If $w = [w_1, w_2, \dots, w_k]$, then **$k$** is the **number of elements in the kernel**.
* Example: If $w = [2, 1, -1]$, then $k = 3$

This is the most common interpretation in deep learning: **kernel size = number of weights**

---

### ✅ 2. **Support of a Function**

In more mathematical terms, the **support** of a function is the region (or domain) where the function is **non-zero**.

So for a **discrete kernel $w$**:

* The **support** is the set of indices $j$ where $w[j] \ne 0$
* In practice, filters have finite support — only a small window of non-zero weights

Example:

```text
w = [0, 0, 1, -1, 0]
support(w) = {3, 4}   (indexing from 1)
```

In deep learning, since kernels are finite and usually dense (few or no zeros), **support size ≈ kernel length**.

---

### 🧠 Summary

| Term             | Meaning in context of convolution                |
| ---------------- | ------------------------------------------------ |
| **k**            | Length (number of elements in the kernel/filter) |
| **Support**      | The region where the kernel is non-zero          |
| **Support size** | Often equal to $k$, unless the kernel is sparse  |
