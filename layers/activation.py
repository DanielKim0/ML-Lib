from base import Layer
from comp.activation import *


class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU Layer"

    def __repr__(self):
        return "ReLULayer()"

    def op(self, X):
        return relu(X)

class LeakyReLULayer(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def __str__(self):
        return f"Leaky ReLU Layer: alpha={self.alpha}"

    def __repr__(self):
        return f"LeakyReLULayer(alpha={self.alpha})"

    def op(self, X):
        return leaky_relu(X, self.alpha)

class PReLULayer(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def __str__(self):
        return f"PReLU Layer: alpha={self.alpha}"

    def __repr__(self):
        return f"PReLULayer(alpha={self.alpha})"

    def op(self, X):
        return prelu(X, self.alpha)

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Sigmoid Layer"

    def __repr__(self):
        return "SigmoidLayer()"    

    def op(self, X):
        return sigmoid(X)

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Tanh Layer"

    def __repr__(self):
        return "TanhLayer()"

    def op(self, X):
        return tanh(X)
