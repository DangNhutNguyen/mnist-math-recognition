import numpy as np
import matplotlib.pyplot as plt

# Load MNIST data
images = np.load('data/mnist_train_images.npy')
labels = np.load('data/mnist_train_labels.npy')

# Randomly select 10 images
indices = np.random.choice(len(images), 10, replace=False)
selected_images = images[indices]
selected_labels = labels[indices]

# Plot the images
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(selected_images[i], cmap='gray')
    ax.set_title(f"Label: {selected_labels[i]}")
    ax.axis('off')
plt.show()
