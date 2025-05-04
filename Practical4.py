# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.datasets import fashion_mnist
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import confusion_matrix
# # Load dataset
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# # Reshape for CNN input and normalize
# X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
# X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# # One-hot encode labels
# y_train_cat = to_categorical(y_train, 10)
# y_test_cat = to_categorical(y_test, 10)
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
#     MaxPooling2D(pool_size=(2,2)),
#     Dropout(0.25),

#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(pool_size=(2,2)),
#     Dropout(0.25),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')  # 10 classes
# ])
# optimizer = Adam(learning_rate=0.001)

# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Train model
# model.fit(X_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.2)
# # Evaluate
# test_loss, test_acc = model.evaluate(X_test, y_test_cat)
# print(f"Test Accuracy: {test_acc*100:.2f}%")

# # Predict classes
# y_pred = np.argmax(model.predict(X_test), axis=1)

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Flatten + Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_test, y_test))

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()
