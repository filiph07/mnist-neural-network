import numpy as np

def load_images(filename):
    with open(filename, "rb") as file:
        file.read(16) # discard the 16-byte metadata header
        raw_pixels = np.frombuffer(file.read(), dtype=np.uint8)
    
    # reshape into 2d grid: (num_images, 784_pixels_per_image)
    images = raw_pixels.reshape(-1, 784)

    # transpose so each column is one image: (784, num_images)
    images_transposed = images.T

    # scale pixel values from [0, 255] down to [0.0, 1.0]
    normalised_images = images_transposed / 255.0

    return normalised_images

def load_labels(filename):
    with open(filename, "rb") as file:
        file.read(8) # discard the 8-byte metadata header
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    
    return labels

def one_hot_encode(labels, num_classes=10):
    num_samples = labels.shape[0]

    # create a grid of all zeros: shape (10, num_samples)
    one_hot = np.zeros((num_classes, num_samples))

    # place a 1.0 at row `label` for each column (sample index)
    one_hot[labels, np.arange(num_samples)] = 1.0

    return one_hot

train_images = load_images("train-images-idx3-ubyte")
train_labels = load_labels("train-labels-idx1-ubyte")