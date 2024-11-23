# **MNIST Math Recognition**

This project demonstrates digit recognition from the MNIST dataset using **pure mathematics**, specifically **Euclidean distance** for similarity measurement. We explore how simple mathematical operations can be used to recognize digits without relying on complex machine learning algorithms.

---

## **Mathematical Explanation**

### **What is Euclidean Distance?**
Euclidean distance is one of the most common ways to measure the "closeness" or "similarity" between two points in a multi-dimensional space. It is the straight-line distance between two points, which in this case are pixel values from two images.

Consider two images \(A\) and \(B\). Each image is represented as a **vector** of pixel values. These images can be thought of as points in a multi-dimensional space where each dimension corresponds to a pixel in the image.

The Euclidean distance between two images \(A\) and \(B\) can be calculated using the following formula:

$\text{Distance}(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}$

Where:
- \(A_i\) is the pixel value at the \(i\)-th position in image \(A\),
- \(B_i\) is the pixel value at the \(i\)-th position in image \(B\),
- \(n\) is the number of pixels in the image (for MNIST, it is typically \(28 \times 28 = 784\) pixels per image).

### **Mathematical Approach in the Code**
In the `recognize_digits.py` script, the **Euclidean distance** is used to compare the **test image** with all the images in the **training dataset**.

For each image in the dataset, the script calculates the Euclidean distance between the test image and the training image. The image with the smallest distance is the one that is most similar to the test image. Thus, it is identified as the predicted label for the test image.

This is a **brute-force approach**: we simply compute the distance between the test image and each image in the dataset, and choose the image that is closest.

### **Labeling the Images**

The MNIST dataset consists of images of handwritten digits (0–9). Each image is associated with a label that represents the digit the image is depicting. For instance:

- An image might have a digit "3" written on it, and the label for that image would be 3.
- The images and labels are typically paired, meaning the dataset contains a collection of **image-label pairs**.

In the context of this project:
- The **label** is simply the digit associated with the image.
- The **image** is a 28x28 pixel array representing the pixel intensities (grayscale values) of the handwritten digit.

### **Process of Recognition**:
1. **Training Dataset**: The dataset consists of images of handwritten digits, where each image is associated with a specific digit label (0–9).
2. **Test Image**: We take a test image (one that we want to recognize) and compute the Euclidean distance between this image and each image in the dataset.
3. **Recognition**: The image that has the smallest Euclidean distance to the test image is considered the most similar, and the label of that image is returned as the **predicted label**.

#### Example:
If we take the first image in the dataset (let's say it's a "3"), we compare it to all other images in the dataset. Suppose the image with the smallest distance is labeled "3" as well, then the **predicted label** is correct.

If the test image is compared to an image labeled "1", and the distance is smaller than any other image's distance, the predicted label will be "1".

---

## **Code Explanation**

### **Step 1: Data Generation (`generate_mnist.py`)**
This script:
- Loads the MNIST dataset from TensorFlow.
- Preprocesses the images by normalizing them to a range of [0, 1] (which is better for similarity comparison).
- Saves the images and their corresponding labels to `.npy` files for later use.

```python
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
```

### **Step 2: Visualization (`visualize_mnist.py`)**
This script:
- Loads the preprocessed MNIST images and labels.
- Randomly selects 10 images and displays them along with their labels for visual inspection.

```python
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
```

### **Step 3: Digit Recognition (`recognize_digits.py`)**
This script:
- Takes a test image and compares it to the entire dataset using Euclidean distance.
- Displays the test image with its actual label and the predicted label.

```python
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
```

### **Mathematical Approach Recap**:
1. **Euclidean Distance** is calculated between the test image and every image in the dataset.
2. The image with the **smallest Euclidean distance** is considered the **closest** match.
3. The label of the closest image is returned as the predicted label.

---

## **How to Run**

### **Prerequisites**
- Python 3.6 or higher
- Required Python packages:
  - `numpy`
  - `tensorflow`
  - `matplotlib`

You can install the required packages using pip:

```bash
pip install numpy tensorflow matplotlib
```

### **Steps to Run the Project**

1. **Clone the repository**:

```bash
git clone https://github.com/dangnhutnguyen/mnist-math-recognition.git
cd mnist-math-recognition
```

2. **Generate MNIST data**:

The first step is to generate the MNIST dataset by running the `generate_mnist.py` script. This will preprocess the images and save them as `.npy` files in the `data` directory.

```bash
python src/generate_mnist.py
```

3. **Visualize MNIST images**:

To visualize a selection of random images from the MNIST dataset, run the `visualize_mnist.py` script. This will display 10 random images with their corresponding labels.

```bash
python src/visualize_mnist.py
```

4. **Run the digit recognition**:

To test the digit recognition, run the `recognize_digits.py` script. This will show a test image and compare it to the training dataset, displaying both the actual and predicted labels.

```bash
python src/recognize_digits.py
```

---

## **Limitations of the Math Approach**

- **Efficiency**: The current approach compares each test image to every image in the dataset, making it computationally expensive. It does not scale well for larger datasets.
- **Accuracy**: While Euclidean distance can work for smaller datasets, it is not as robust or accurate as machine learning models like **Convolutional Neural Networks (CNNs)**, which are specifically designed for image recognition tasks.
- **Feature Extraction**: The Euclidean distance approach compares raw pixel values, which can be affected by minor distortions or changes in image orientation.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
