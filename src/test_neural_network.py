import idx2numpy
import numpy as np

images = idx2numpy.convert_from_file("src/t10k-images.idx3-ubyte")
labels = idx2numpy.convert_from_file("src/t10k-labels.idx1-ubyte")

model = np.load("trained_model.npz")
W_hidden = model["W_hidden"]
b_hidden = model["b_hidden"]
W_output = model["W_output"]
b_output = model["b_output"]

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

correct = 0
total = len(images)

for image, label in zip(images, labels):
    x = np.array(image).reshape(-1, 1) / 255.0

    # Forward pass
    z_hidden = W_hidden @ x + b_hidden
    a_hidden = relu(z_hidden)

    z_output = W_output @ a_hidden + b_output
    y_pred = softmax(z_output)

    predicted_label = np.argmax(y_pred)
    
    if predicted_label == label:
        correct += 1

accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
