from base import Layer
from comp.activation import *

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def op(self, X):
        return relu(X)

class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def op(self, X):
        return leaky_relu(X, self.alpha)

class PReLU(Layer):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def op(self, X):
        return prelu(X, self.alpha)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def op(self, X):
        return sigmoid(X)

class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def op(self, X):
        return tanh(X)
