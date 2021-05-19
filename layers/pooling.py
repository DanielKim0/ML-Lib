import tensorflow as tf
from comp.pooling import pooling2d
from comp.convolution import get_conv_size
from .base import Layer


class PoolingLayer(Layer):
    def __init__(self, size, stride, padding="none", mode="max"):
        super().__init__()
        self.size = [size, size]
        self.stride = stride
        self.padding = padding
        self.mode = mode

    def __str__(self):
        s = "Pooling Layer:\n"
        s += f"size = {self.size}\n"
        s += f"stride = {self.stride}\n"
        s += f"padding = {self.padding}\n"
        s += f"mode = {self.mode}\n"
        return s

    def __repr__(self):
        return f"PoolingLayer(size={self.size}, stride={self.stride}, padding={self.padding}, mode={self.mode})"

    def set_dims(self, inp):
        self.inp = inp
        self.out = get_conv_size(inp, self.size, self.stride, self.padding)

    def op(self, X):
        return pooling2d(X, self.size, self.stride, self.padding, self.mode)
