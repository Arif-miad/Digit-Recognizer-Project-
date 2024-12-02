# Digit Recognizer Project 🧮✍️  

Welcome to the **Digit Recognizer** project! This project demonstrates the process of building a machine learning model to classify handwritten digits (0-9) using the popular MNIST dataset. Through this project, we employ techniques such as data preparation, model training, data augmentation, and visualization to achieve high accuracy.

---

## 🚀 **Project Workflow**  
### 1. **Import Libraries**  
We import essential libraries for data manipulation, visualization, and model building.  
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### 2. **Load Dataset**  
The dataset is loaded and preprocessed for analysis. The MNIST dataset contains images of handwritten digits with corresponding labels.  
```python
from tensorflow.keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

### 3. **Data Preparation**  
We normalize the pixel values to the range [0, 1] and reshape the images for compatibility with the neural network. Labels are converted to one-hot encoding.  
```python
# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for neural network input
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
```

### 4. **Data Splitting**  
We split the training data into training and validation sets.  
```python
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```

### 5. **Model Definition**  
We define a Convolutional Neural Network (CNN) for image classification.  
```python
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 6. **Train the Model**  
We train the model using the training data while validating it on the validation set.  
```python
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=64)
```

### 7. **Data Augmentation**  
To improve generalization, we apply data augmentation techniques like rotation and zoom.  
```python
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(X_train)

# Train with augmented data
history_augmented = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                              validation_data=(X_val, y_val),
                              epochs=10)
```

### 8. **Evaluate and Test Prediction**  
We evaluate the model's performance on the test set and predict results.  
```python
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
predictions = model.predict(X_test)
```

### 9. **Visualize Results**  
We visualize model performance and sample predictions.  
```python
# Plot accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

# Display some test predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.show()
```

---

## 📂 **Project Structure**  
- **`data/`**: Contains dataset files.  
- **`notebooks/`**: Jupyter notebooks for exploration and model building.  
- **`models/`**: Saved models after training.  
- **`results/`**: Contains plots and results of predictions.  

---

## 🚀 **How to Run the Project**  
1. Clone this repository.  
2. Install required dependencies using `pip install -r requirements.txt`.  
3. Run the Jupyter notebook or Python scripts to train and evaluate the model.  

---

## 💡 **Key Insights**  
- Data normalization and augmentation significantly improve model performance.  
- Simplicity in model design can achieve high accuracy on MNIST-like datasets.  

---

### 🌟 **Contributions and Feedback**  
Feel free to fork, explore, and suggest improvements to this project. If you like this work, don’t forget to give it a ⭐️ on GitHub!  

---  

### 📌 **References**  
- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)  

---

