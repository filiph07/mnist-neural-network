import numpy as np

def load_images(filename):
    with open(filename, "rb") as file:
        file.read(16) # discard the 16-byte metadata header
        raw_pixels = np.frombuffer(file.read(), dtype=np.uint8)
    
    # 1. reshape into 2d grid: (num_images, 784_pixels_per_image)
    images = raw_pixels.reshape(-1, 784)

    # 2. transpose so each column is one image: (784, num_images)
    images_transposed = images.T

    # 3. scale pixel values from [0, 255] down to [0.0, 1.0]
    normalised_images = images_transposed / 255.0

    return normalised_images

train_images = load_images("train-images-idx3-ubyte")
print("Shape:", train_images.shape)
print("Pixel range:", train_images.min(), "to", train_images.max())