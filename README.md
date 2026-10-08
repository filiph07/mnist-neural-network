# MNIST Neural Network from Scratch in NumPy

A two-layer feedforward neural network built entirely from mathematical first principles using only Python and NumPy to classify handwritten digits from the MNIST dataset.

No PyTorch, no TensorFlow, no scikit-learn. I wanted to understand what actually happens under the hood of deep learning before leaning on high-level frameworks—deriving the calculus by hand, reading the raw binary files, and building the entire pipeline using vectorized linear algebra.

---

## Results at a Glance

| Metric | Value |
|---|---:|
| **Training Accuracy** | ~92% (500 iterations) |
| **Test Accuracy** | ~91% |
| **Test Set** | 10,000 unseen digits (`t10kk`) |
| **Input Layer** | 784 features ($28 \times 28$ grayscale pixels) |
| **Hidden Layer** | 128 units (ReLU activation) |
| **Output Layer** | 10 units (Softmax distribution, digits 0–9) |

Hitting ~91% on unseen test digits confirms the network learned generalised features instead of just memorising training pixels.

---

## Architecture & Pipeline

```text
Input (784, m) ──▰ Hidden Layer (128, m) [ReLU] ──▰ Output Layer (10, m) [Softmax]
```

The pipeline uses a column-oriented format where each column is an individual sample across $m$ images. This aligns matrix operations directly with standard linear algebra notation ($Z = WX + b$).

### 1. Forward Propagation
* **Hidden Layer:**
  $$Z_1 = W_1 X + b_1$$
  $$A_1 = \max(0, Z_1)$
* **Output Layer:**
  $$Z_2 = W_2 A_1 + b_2$$
  $$A_2 = \textsoftmax}(Z_2)$$**Numerically Stable Softmax:** Exponentiating large logits directly leads to floating-point overflow (`NaN` errors). To prevent this, I subtract the column-wise maximum before exponentiating:
  $$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\��j e^{z_j - \max(z)}}d%

### 2. Manual Backpropagation
Rather than relying on autograd engines, gradients were derived by hand using the chain rule on categorical cross-entropy loss:
* **Output Error:** $dZ_2 = A_2 - Y$ (where $Y$ is the one-hot encoded ground truth)
* **Hidden Error:** $dZ_1 = (W_2^T dZ_2) \odot (Z_1 > 0)$, gating the gradient through the derivative of ReLU
* **Parameter Updates:**
  $$dW_2 = \frac{1}{m} dZ_2 A_1^T, \quad $db_2 = \frac{1}{m} \sum_{\text{cols}} dZ_2$$
  $$dW_1 = \frac{1}{m} dZ_1 X^T, \quad $db_1 = \frac{1}{m} \sum_{\text{cols}} dZ_1$$

### 3. He (Kaiming) Initialisation
Standard Gaussian sampling with unscaled variances causes activations to vanish or explode when paired with ReLU. Weights are scaled using He initialisation based on the layer's fan-in:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan_in}}}\right)$$

---

## Custom Binary Parsing (No Dataset Helpers)

Instead of using `torchvision` or `keras.datasets` to download pre-cleaned arrays, this project reads the official raw `.idx` binary files directly from disk:
1. Strips the 16-byte image and 8-byte label file headers.
2. Unpacks unsigned 8-bit integers into NumPy arrays using `np.frombuffer`.
3. Flattens each $28 \times 28$ image into a 784-element column vector and scales byte values from $[0, 255]$ into $[0.0, 1.0]$.

---

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install numpy
   ```

2. **Add dataset files:**
   Place the four extracted MNIST byte files in the project root.

3. **Run training and evaluation:**
   ```bash
   python main.py
   ```

---

## Why Build It This Way?

It is easy to write a few lines of PyTorch and reach high accuracy without understanding how the internal mechanics function. Building this from scratch made me work through:
* How vectorised broadcasting replaces slow Python loops with BLAS-level array execution.
* Why subtracting max values in Softmax is essential to avoid floating-point overflow.
* How the chain rule translates into matrix transpositions and dot products in code.

---

## Tech Stack
* **Language:** Python
* **Library:** NumPy
* **Tools:** Git, GitHub
