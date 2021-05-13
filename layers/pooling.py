import tensorflow as tf
from base import Layer
from comp.pooling import *

class Pooling(Layer):
    def __init__(self, size, mode="max"):
        super().__init__()
        self.size = size
        self.mode = mode

    def set_dims(self, inp):
        self.inp = inp
        self.out = [inp[i] - self.size[i] for i in range(len(size))]

    def op(self, X):
        return pooling2d(X, self.size, self.mode)
