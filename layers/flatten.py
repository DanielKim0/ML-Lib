import math
import tensorflow as tf
from base import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def op(self, X):
        return tf.reshape(X, [X.shape[0], math.sum(X.shape[1:])])
