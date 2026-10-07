import numpy as np

def load_images(filename):
    with open(filename, "rb") as file:
        file.read(16) # skip 16-byte header containing magic number and dims
        raw_pixels = np.frombuffer(file.read(), dtype=np.uint8)
    
    # reshape to 2D array where each row is an image
    images = raw_pixels.reshape(-1, 784)

    # transpose so images are stored as columns: shape (784, num_samples)
    images_transposed = images.T

    # normalise pixel values to range [0.0, 1.0]
    normalised_images = images_transposed / 255.0

    return normalised_images

def load_labels(filename):
    with open(filename, "rb") as file:
        file.read(8) # skip 8-byte header containing magic number and count
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    
    return labels

def one_hot_encode(labels, num_classes=10):
    num_samples = labels.shape[0]

    # initialise target matrix with zeros
    one_hot = np.zeros((num_classes, num_samples))

    # set the row corresponding to the label to 1.0 for each column
    one_hot[labels, np.arange(num_samples)] = 1.0

    return one_hot

def init_params(input_size=784, hidden_size=128, output_size=10):
    # He initialisation for hidden layer (var = 2 / fan_in)
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
    b1 = np.zeros((hidden_size, 1))

    # He initialisation for output layer
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((output_size, 1))

    return W1, b1, W2, b2

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # subtract column max to prevent numerical overflow in exp
    shifted_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shifted_Z)

    # normalise by column sums so output values form a probability distribution
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

train_images = load_images("train-images-idx3-ubyte")
train_labels = load_labels("train-labels-idx1-ubyte")
train_labels_encoded = one_hot_encode(train_labels)
W1, b1, W2, b2 = init_params()