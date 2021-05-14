from math import prod
import tensorflow as tf
from .base import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        
    def set_dims(self, inp):
        self.inp = inp
        self.out = [prod(inp)]

    def op(self, X):
        return tf.reshape(X, [X.shape[0], prod(X.shape[1:])])
