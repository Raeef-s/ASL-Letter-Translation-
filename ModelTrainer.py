from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




print("Numpy Version: " + np.__version__)
#print("Matplotlib Version: " + plt.__version__)
print("Tensorflow Version: " + tf.__version__)

data = pd.read_csv("MediapipePredictions.csv")

print(data.head())

# Drops all empty categories
df = data.dropna()

# Prints out value counts
classifications = df.Classification.value_counts()
print(classifications)

def compute_class_weights(classes):

    # Dictionary of all classifications, and how often they occur
    counts = Counter(classes)
    # Number of samples
    total_samples = len(classes)
    # Number of classifications
    num_classes = len(counts)

    # Creates another dictionary with the weight for each corresponding class
    # weight = total_samples / (num_classes * count for that class)
    class_weights = {cls: total_samples / (num_classes * count)
                     for cls, count in counts.items()}
    return class_weights



# Splits dataframe into training data DataFrame, and classification Series
X = df.drop("Classification", axis=1)
y = np.array(df["Classification"])

# Splits data into training and temporary sets (80% training, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)
# Splits temporary set into CV set and Test set (50:50 CV and Test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1234, stratify=y_temp)

print("Training set class distribution:", Counter(y_train))
print("Validation set class distribution:", Counter(y_val))
print("Test set class distribution:", Counter(y_test))

#
class_weights = compute_class_weights(y_train)

print("Class Weights:", class_weights)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_val = encoder.fit_transform(y_val)
y_test = encoder.fit_transform(y_test)
print("Label mapping:", encoder.classes_)


model = Sequential([
    layers.Dense(128, activation='relu', input_shape=[105]),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(28, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=30, class_weight=class_weights, validation_data=(X_val, y_val))

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=128)
print(predictions)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

model.save("MediapipeFFN.keras")


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()