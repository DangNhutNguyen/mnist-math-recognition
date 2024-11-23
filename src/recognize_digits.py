import numpy as np
import matplotlib.pyplot as plt

# Function to calculate similarity (Euclidean distance)
def recognize_digit(image, dataset):
    distances = [np.linalg.norm(image - img) for img in dataset]
    return np.argmin(distances)

# Load the generated dataset
images = np.load('data/mnist_train_images.npy')
labels = np.load('data/mnist_train_labels.npy')

# Select the first image for demonstration
test_image = images[0]
actual_label = labels[0]

# Recognize the digit
predicted_index = recognize_digit(test_image, images)
predicted_label = labels[predicted_index]

# Display the result
plt.imshow(test_image, cmap='gray')
plt.title(f"Actual Label: {actual_label} | Predicted Label: {predicted_label}")
plt.axis('off')
plt.show()
