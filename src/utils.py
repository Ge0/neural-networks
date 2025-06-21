import numpy as np
import math

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
        return (
            sum(
                entry * weight
                for entry, weight in zip(self.entries, self.weights)
            )
            + self.bias
        )

    def get_result(self):
        if not hasattr(self, "_cached_result"):
            self._cached_result = self.activation_function(self.output())
        return self._cached_result

    def reset_cache(self):
        if hasattr(self, "_cached_result"):
            del self._cached_result


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    return -sum(
        y_t * math.log(max(y_p, epsilon)) for y_t, y_p in zip(y_true, y_pred)
    )
