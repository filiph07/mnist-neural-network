# MNIST from Scratch in NumPy

A 2-layer neural network built from scratch using only Python and NumPy to classify handwritten digits from the MNIST dataset.

I wanted to understand how backpropagation, matrix operations, and gradient descent actually work under the hood without relying on black-box frameworks like PyTorch or TensorFlow.

---

## Results

| Metric | Result |
|---|---:|
| **Training Accuracy** | ~92% (500 iterations) |
| **Test Accuracy** | ~91% |
| **Test Set** | 10,000 unseen digits (t10k) |
| **Architecture** | 784 -> 128 (ReLU) -> 10 (Softmax) |

Hitting ~91% test accuracy confirms the network learns generalised digit shapes rather than just memorising the training set.

---

## What It Does

1. **Custom Data Loader**: Reads raw MNIST `.idx` binary files directly using `np.frombuffer`, normalising pixel intensities to [0.0, 1.0].
2. **Vectorised Linear Algebra**: Formats samples as columns (`784, m`) so matrix operations follow standard maths notation ($Z = WX + b$).
3. **Numerically Stable Activations**: Uses ReLU for hidden units and a shift-invariant Softmax to prevent floating-point overflow (`NaN` crashes).
4. **Manual Backprop**: Gradients are derived and computed by hand using matrix calculus, updated via gradient descent.
5. **He Initialisation**: Scales starting weights by incoming fan-in to keep activation variance steady across layers.

---

## Setup and Running

1. Install NumPy:
   pip install numpy

2. Drop the 4 uncompressed MNIST files into the project folder:
   * train-images-idx3-ubyte
   * train-labels-idx1-ubyte
   * t10k-images-idx3-ubyte
   * t10k-labels-idx1-ubyte

3. Run the script:
   python main.py
