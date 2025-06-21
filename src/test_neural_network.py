import idx2numpy
import numpy as np
import random
import math

images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")

INPUT_SIZE = 28 * 28

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class EntryNeuron:
    def __init__(self, value):
        self.value = value

    def output(self):
        return self.value / 255.0
    
    def get_result(self):
        return self.output()


class Perceptron:
    def __init__(self, entries, weights, bias, activation_function):
        self.entries = entries
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function

    def output(self):
        return sum(
            entry * weight
            for entry, weight in zip(self.entries, self.weights)
        ) + self.bias

    def get_result(self):
        return self.activation_function(self.output())


print("[+] Initialize the hidden layer.")
hidden_layer = [
    Perceptron(
        entries=[0] * INPUT_SIZE,
        weights=[0] * INPUT_SIZE,
        bias=0,
        activation_function=lambda x: max(0, x),
    )
    for _ in range(128)
]

for perceptron in hidden_layer:
    perceptron.weights = [
        random.gauss(0, math.sqrt(2 / INPUT_SIZE)) for _ in range(28 * 28)
    ]

print("[+] Initialize the output layer.")
output_layer = [
    Perceptron(
        entries=[0]*128,
        weights=[0]*128,
        bias=0,
        activation_function=lambda output: output
    ) for _ in range(10)
]

for perceptron in output_layer:
    perceptron.weights = [
        random.gauss(0, math.sqrt(2 / len(hidden_layer)))
        for _ in range(len(hidden_layer))
    ]
    perceptron.bias = 0

correct = 0
total = 0
for image, label in zip(images[:100], labels[:100]):
    print(f"[+] Initialize input layer with new image.")
    flat_image = np.array(image).flatten().tolist()
    input_layer = [EntryNeuron(value=pixel) for pixel in flat_image]
    
    # Compute the hidden layer.
    print(f"[+] First pass: hidden layer.")
    for perceptron in hidden_layer:
        perceptron.entries = [neuron.get_result() for neuron in input_layer]
    
    # Compute the output layer.
    print(f"[+] Second pass: output layer.")
    for perceptron in output_layer:
        perceptron.entries = [neuron.get_result() for neuron in hidden_layer]
    
    # Get the outputs.
    outputs = [perceptron.get_result() for perceptron in output_layer]
    
    # Softmax.
    result = softmax(outputs)
    
    print(f"[+] Result: {result}")
    prediction = np.argmax(result)
    print(f" Prediction: {prediction} — Correct: {label}")
    if prediction == label:
        correct += 1
    total += 1

print(f"Accuracy: {(correct/total):.2%}")