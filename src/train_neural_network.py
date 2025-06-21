import idx2numpy
import numpy as np


images = idx2numpy.convert_from_file("src/train-images.idx3-ubyte")
labels = idx2numpy.convert_from_file("src/train-labels.idx1-ubyte")

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 10

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    return -np.sum(y_true * np.log(np.clip(y_pred, epsilon, 1. - epsilon)))



W_hidden = np.random.randn(HIDDEN_SIZE, INPUT_SIZE) * np.sqrt(2. / INPUT_SIZE)
b_hidden = np.zeros((HIDDEN_SIZE, 1))

W_output = np.random.randn(OUTPUT_SIZE, HIDDEN_SIZE) * np.sqrt(1. / HIDDEN_SIZE)
b_output = np.zeros((OUTPUT_SIZE, 1))


for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    for image, label in zip(images, labels):
        x = np.array(image).flatten().reshape(-1, 1) / 255.0
        z_hidden = W_hidden @ x + b_hidden
        a_hidden = relu(z_hidden)

        z_output = W_output @ a_hidden + b_output
        y_pred = softmax(z_output)

        y_true = np.zeros((OUTPUT_SIZE, 1))
        y_true[label] = 1

        loss = cross_entropy(y_true, y_pred)
        total_loss += loss
        if np.argmax(y_pred) == label:
            correct += 1

        dz_output = y_pred - y_true                             # (10, 1)
        dW_output = dz_output @ a_hidden.T                      # (10, 128)
        db_output = dz_output                                   # (10, 1)

        da_hidden = W_output.T @ dz_output                      # (128, 1)
        dz_hidden = da_hidden * relu_derivative(z_hidden)       # (128, 1)
        dW_hidden = dz_hidden @ x.reshape(1, -1)                # (128, 784)
        db_hidden = dz_hidden                                   # (128, 1)

        W_output -= LEARNING_RATE * dW_output
        b_output -= LEARNING_RATE * db_output

        W_hidden -= LEARNING_RATE * dW_hidden
        b_hidden -= LEARNING_RATE * db_hidden

    accuracy = correct / len(images)
    avg_loss = total_loss / len(images)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")

np.savez("trained_model.npz", 
         W_hidden=W_hidden, 
         b_hidden=b_hidden, 
         W_output=W_output, 
         b_output=b_output)
print("Model saved!")
