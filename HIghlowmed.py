# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load the dataset
#url = "https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-PricePrediction/master/USA_Housing.csv"
df = pd.read_csv('USA_Housing.csv')

# Select features and target
X = df.drop(columns=["Price", "Address"])  # Drop target and non-numeric column
y = df["Price"]

# Create categorical target labels: Low, Medium, High
price_labels = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])
le = LabelEncoder()
y_encoded = le.fit_transform(price_labels)
y_categorical = to_categorical(y_encoded)  # One-hot encoding

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input for Conv1D: (samples, features, 1)
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_categorical, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_cnn.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))  # 3 categories: Low, Medium, High

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
