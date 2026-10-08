# MNIST Neural Network From Scratch

A two-layer neural network implemented from scratch using Python and NumPy to
classify handwritten digits from the MNIST dataset.

The project focuses on understanding the mathematics and implementation behind
neural networks rather than relying on machine learning frameworks.

## Results

| Metric | Result |
|---|---:|
| Training accuracy | ~92% |
| Test accuracy | ~91% |
| Test set | 10,000 unseen images |
| Input size | 784 |
| Hidden units | 128 |
| Output classes | 10 |

The model achieves approximately 91% accuracy on the 10,000-image MNIST test
set without using machine learning libraries such as PyTorch or TensorFlow.

## Architecture

The network consists of:

```text
784 input features
       ↓
128 hidden units
       ↓
10 output classes