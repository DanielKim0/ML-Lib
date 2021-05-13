import tensorflow as tf
from base import Layer
from comp.pooling import *
from comp.convolution import get_conv_size

class Pooling(Layer):
    def __init__(self, size, stride, padding, mode="max"):
        super().__init__()
        self.size = size
        self.stride = stride
        self.padding = padding
        self.mode = mode

    def set_dims(self, inp):
        self.inp = inp
        self.out = get_conv_size(inp, self.kernel_size, self.stride, self.padding)

    def op(self, X):
        return pooling2d(X, self.size, self.mode)
