# Simple Neural Network from Scratch in NumPy

This project implements a fully connected neural network from scratch using only NumPy. It classifies handwritten digits from the MNIST dataset and is designed to be educational, transparent, and easy to follow.

## 📚 Description

The network architecture consists of:
- An input layer of 784 values (28x28 grayscale pixels)
- One hidden layer with 128 neurons using ReLU activation
- An output layer with 10 neurons (one for each digit), using softmax

The model is trained using stochastic gradient descent and cross-entropy loss.

## 📁 Files

- `train_neural_network.py`: Trains the model on the MNIST training set and saves the learned parameters to a `.npz` file.
- `test_neural_network.py`: Loads the trained model and evaluates its accuracy on the MNIST test set.

## ⚙️ Requirements

- Python 3
- Libraries: `numpy`, `idx2numpy`


Create a virtual environment if needed.

```bash
python -m virtualenv venv
source venv/bin/activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

## 🚀 Training

To train the model, run from the project’s root:

```bash
PYTHONPATH=. python -m src.train_neural_network
Epoch 1: Loss=0.2156, Accuracy=93.53%
Epoch 2: Loss=0.0960, Accuracy=97.16%
Epoch 3: Loss=0.0662, Accuracy=98.07%
Epoch 4: Loss=0.0499, Accuracy=98.49%
Epoch 5: Loss=0.0372, Accuracy=98.93%
Epoch 6: Loss=0.0279, Accuracy=99.19%
Epoch 7: Loss=0.0221, Accuracy=99.40%
Epoch 8: Loss=0.0169, Accuracy=99.58%
Epoch 9: Loss=0.0134, Accuracy=99.68%
Epoch 10: Loss=0.0103, Accuracy=99.77%
Model saved!
```

You should have a `train_model.npz` file appearing in the current directory. This is your trained model / neural network.

## 🔍 Testing

Now, to evaluate the trained model on the test set, run:

```bash
PYTHONPATH=. python -m src.test_neural_network
Test Accuracy: 97.56%
```


## 🧠 Why This Project?

This project is meant to help you understand:

- Forward propagation
- Backpropagation
- Manual parameter updates without relying on machine learning libraries

It's a clean and simple way to see what's really going on inside a neural network.

Let me know if you'd like to add GitHub-specific sections like badges, license info, or a roadmap!