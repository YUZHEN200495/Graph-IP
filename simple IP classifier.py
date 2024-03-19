import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create a simple IP address classification model

# Input layer, an IP address typically consists of 4 parts
model = keras.Sequential([
    keras.layers.Input(shape=(4,), name="ip_address"),
    # Hidden layer 1
    keras.layers.Dense(64, activation='relu'),
    # Hidden layer 2
    keras.layers.Dense(32, activation='relu'),
    # Output layer, assuming there are 3 categories
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Generate more example IP address data and labels
num_samples = 5000  # Assume there are 5000 examples
ip_addresses = np.random.randint(0, 256, size=(num_samples, 4))  # Randomly generate IP addresses
labels = np.random.randint(0, 3, size=num_samples)  # Randomly generate category labels

# Split into training and testing sets
split_ratio = 0.8  # 80% of the data for training, 20% for testing
split_index = int(num_samples * split_ratio)
train_ip = ip_addresses[:split_index]
train_labels = labels[:split_index]
test_ip = ip_addresses[split_index:]
test_labels = labels[split_index:]

# Train the model
model.fit(train_ip, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_ip, test_labels)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Use the model for IP address classification
new_ip = np.array([[192, 168, 0, 1]])  # Example IP address
predicted_label = model.predict(new_ip)
predicted_class = np.argmax(predicted_label)

# Print the prediction result
print("Predicted Label:", predicted_class)
