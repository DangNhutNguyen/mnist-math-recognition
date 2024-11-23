import numpy as np
from tensorflow.keras.datasets import mnist

# Load MNIST data
(train_images, train_labels), (_, _) = mnist.load_data()

# Normalize images to the range [0, 1]
train_images = train_images.astype('float32') / 255.0

# Save preprocessed data
np.save('data/mnist_train_images.npy', train_images)
np.save('data/mnist_train_labels.npy', train_labels)

print("MNIST training data saved successfully!")
