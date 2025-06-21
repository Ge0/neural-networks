import idx2numpy
import numpy as np
import random
import math
import json

from .utils import Perceptron, EntryNeuron, cross_entropy, softmax

images = idx2numpy.convert_from_file("src/train-images.idx3-ubyte")
labels = idx2numpy.convert_from_file("src/train-labels.idx1-ubyte")

INPUT_SIZE = 28 * 28
LEARNING_RATE = 0.01


def save_model(hidden_layer, output_layer, filename="src/model.json"):
    model_data = {
        "hidden_layer": [
            {"weights": neuron.weights, "bias": neuron.bias}
            for neuron in hidden_layer
        ],
        "output_layer": [
            {"weights": neuron.weights, "bias": neuron.bias}
            for neuron in output_layer
        ]
    }
    with open(filename, "w") as stream:
        json.dump(model_data, stream)




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
        entries=[0] * 128,
        weights=[0] * 128,
        bias=0,
        activation_function=lambda output: output,
    )
    for _ in range(10)
]

for perceptron in output_layer:
    perceptron.weights = [
        random.gauss(0, math.sqrt(2 / len(hidden_layer)))
        for _ in range(len(hidden_layer))
    ]
    perceptron.bias = 0


for epoch in range(10):
    print(f"[!] EPOCH {epoch}")
    correct = 0
    total = 0
    for image, label in zip(images, labels):
        print(f"[+] Initialize input layer with new image.")
        flat_image = np.array(image).flatten().tolist()
        input_layer = [EntryNeuron(value=pixel) for pixel in flat_image]

        # Compute the hidden layer.
        print(f"[+] First pass: hidden layer.")
        for perceptron in hidden_layer:
            perceptron.entries = [
                neuron.get_result() for neuron in input_layer
            ]

        # Compute the output layer.
        print(f"[+] Second pass: output layer.")
        for perceptron in output_layer:
            perceptron.entries = [
                neuron.get_result() for neuron in hidden_layer
            ]

        # Get the outputs.
        outputs = [perceptron.get_result() for perceptron in output_layer]

        # Softmax.
        y_pred = softmax(outputs)

        # print(f"[+] Result: {y_pred}")
        prediction = np.argmax(y_pred)
        print(f" Prediction: {prediction} — Correct: {label}")

        y_true = [1 if i == label else 0 for i in range(10)]

        loss = cross_entropy(y_true, y_pred)
        print(f"Loss: {loss}")

        output_deltas = [y_p - y_t for y_p, y_t in zip(y_pred, y_true)]

        if prediction == label:
            correct += 1
        total += 1

        print("Adjust the weights and the bias.")
        # Update the output layer.
        for j, neuron in enumerate(output_layer):
            new_weights = list()
            for i, weight in enumerate(neuron.weights):
                new_weights.append(
                    weight
                    - LEARNING_RATE
                    * output_deltas[j]
                    * hidden_layer[i].get_result()
                )
            neuron.bias -= LEARNING_RATE * output_deltas[j]
            neuron.weights = new_weights

        # Retropropagate.
        hidden_deltas = list()
        for i, hidden_neuron in enumerate(hidden_layer):
            z_i = hidden_neuron.output()
            relu_derivative = 1 if z_i > 0 else 0
            backprop_error = sum(
                output_deltas[j] * output_layer[j].weights[i]
                for j in range(len(output_layer))
            )
            delta = relu_derivative * backprop_error
            hidden_deltas.append(delta)

        for i, hidden_neuron in enumerate(hidden_layer):
            delta = hidden_deltas[i]
            for k in range(len(hidden_neuron.weights)):
                hidden_neuron.weights[k] -= (
                    LEARNING_RATE * delta * input_layer[k].output()
                )
            hidden_neuron.bias -= LEARNING_RATE * delta

        # Reset caches for next image
        for neuron in hidden_layer + output_layer:
            if hasattr(neuron, "reset_cache"):
                neuron.reset_cache()

    print(
        f"Epoch {epoch}: Total: {total} – "
        f"Correct: {correct} — Accuracy: {(correct/total):.2%}"
    )

save_model(hidden_layer=hidden_layer, output_layer=output_layer)
print("[!] Model saved!")