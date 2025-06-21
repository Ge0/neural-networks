import idx2numpy
import numpy as np
from .utils import Perceptron, EntryNeuron, softmax
import json

images = idx2numpy.convert_from_file("src/t10k-images.idx3-ubyte")
labels = idx2numpy.convert_from_file("src/t10k-labels.idx1-ubyte")

INPUT_SIZE = 28 * 28


def load_model():
    with open("src/model.json") as stream:
        model = json.load(stream)
        hidden_layer = [
            Perceptron(
                entries=[0] * INPUT_SIZE,
                weights=neuron["weights"],
                bias=neuron["bias"],
                activation_function=lambda x: max(0, x)
            ) for neuron in model["hidden_layer"]
        ]
        output_layer = [
            Perceptron(
                entries=[0] * INPUT_SIZE,
                weights=neuron["weights"],
                bias=neuron["bias"],
                activation_function=lambda output: output
            ) for neuron in model["output_layer"]
        ]
        return hidden_layer, output_layer
    raise Exception("Model not found.")

hidden_layer, output_layer = load_model()

correct = 0
total = 0
for image, label in zip(images[:100], labels[:100]):
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
    if prediction == label:
        correct += 1
    total += 1
print(f"Total: {total} – Correct: {correct} — Ratio: {correct/total:.2%}")