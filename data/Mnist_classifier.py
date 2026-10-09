# %% [markdown]
# # Handwritten Digit Classifier (MNIST)
# Deep neural network (TensorFlow/Keras) with Batch Normalization and Dropout.
#

# %% 1. Imports & setup
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, Input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

OUT_DIR = "outputs"          # plots are saved here as well as shown
os.makedirs(OUT_DIR, exist_ok=True)
print("TensorFlow version:", tf.__version__)

# %% 2. Dataset loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train:", x_train.shape, "| y_train:", y_train.shape)
print("x_test: ", x_test.shape, "| y_test: ", y_test.shape)

# Preview a few samples
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.axis("off")
plt.suptitle("Sample MNIST training images")
plt.savefig(f"{OUT_DIR}/samples.png", dpi=150, bbox_inches="tight")
plt.show()

# %% 3. Preprocessing
# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 -> 784 for the dense network
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# Keep integer labels for reports; one-hot labels for training
y_test_labels = y_test.copy()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# %% 4. Model architecture (3 hidden layers + BatchNorm + Dropout)
model = Sequential([
    Input(shape=(784,)),

    Dense(512), BatchNormalization(), Activation("relu"), Dropout(0.3),
    Dense(256), BatchNormalization(), Activation("relu"), Dropout(0.3),
    Dense(128), BatchNormalization(), Activation("relu"), Dropout(0.2),

    Dense(10, activation="softmax"),
])
model.summary()

# %% 5. Compile & train
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    verbose=1,
)

# %% 6. Visualization: training vs. validation accuracy & loss
epochs = range(1, len(history.history["accuracy"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(epochs, history.history["accuracy"], "o-", label="Training")
axes[0].plot(epochs, history.history["val_accuracy"], "s-", label="Validation")
axes[0].set(title="Training vs. Validation Accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(epochs, history.history["loss"], "o-", label="Training")
axes[1].plot(epochs, history.history["val_loss"], "s-", label="Validation")
axes[1].set(title="Training vs. Validation Loss", xlabel="Epoch", ylabel="Loss")
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=150)
plt.show()

# %% 7. Evaluation on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print(classification_report(y_test_labels, y_pred, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% 8. Misclassified examples
wrong = np.where(y_pred != y_test_labels)[0]
print(f"Misclassified: {len(wrong)} of {len(y_test_labels)}")

plt.figure(figsize=(12, 3))
for i, idx in enumerate(wrong[:10]):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"T:{y_test_labels[idx]}\nP:{y_pred[idx]}", fontsize=9)
    plt.axis("off")
plt.suptitle("Misclassified examples (T = true, P = predicted)")
plt.savefig(f"{OUT_DIR}/misclassified.png", dpi=150, bbox_inches="tight")
plt.show()

# %% 9. Save the trained model
model.save(f"{OUT_DIR}/mnist_model.keras")
print(f"Model saved to {OUT_DIR}/mnist_model.keras")