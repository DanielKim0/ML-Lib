import tensorflow as tf
from comp.convolution import conv2d
from comp.param_init import normal
from .core import CoreLayer


class Conv2DLayer(CoreLayer):
    def __init__(self, filters, kernel_size, stride=(1,1), padding="valid", act=None, reg=None, param=None):
        super().__init__(act, reg, param)
        self.filters = filters
        self.kernel_size = [kernel_size, kernel_size] # int that creates a square
        self.padding = padding
        self.stride = stride

    def __str__(self):
        s = "2D Convolution Layer\n"
        if self.initialized:
            s += "Currently initialized\n"
            s += f"inp: {self.inp}\n"
            s += f"out: {self.out}\n"
            s += f"w.shape: {self.w.shape}\n"
        else:
            s += "Currently not initialized\n"
        s += f"filters: {self.filters}\n"
        s += f"kernel_size: {self.kernel_size}\n"
        s += f"stride: {self.kernel_size}\n"
        s += f"padding: {self.padding}\n"
        s += f"act: {self.act}\n"
        s += f"reg: {self.reg}\n"
        s += f"param: {self.param}\n"
        return s

    def __repr__(self):
        if self.initialized:
            s = f"Conv2DLayer(initialized={False}, filters={self.filters}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, act={self.act}, reg={self.reg}, param={self.param})"
        else:
            s = f"Conv2DLayer(initialized={True}, inp={self.inp}, out={self.out}, w.shape={self.w.shape}, filters={self.filters}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, act={self.act}, reg={self.reg}, param={self.param})"
        return s

    def validate(self):
        if self.filters <= 0 or not isinstance(self.filters, int):
            raise ValueError("")
        for dim in self.kernel_size:
            if dim <= 0 or not isinstance(dim, int):
                raise ValueError("")
        for dim in self.stride:
            if dim <= 0 or not isinstance(dim, int):
                raise ValueError("")
        if self.padding not in ["none", "valid", "same", "full"]:
            raise ValueError("")

    def set_dims(self, inp):
        self.inp = inp
        self.kernel_size = self.kernel_size + [inp[2], self.filters]
        self.out = get_conv_size(inp, self.kernel_size, self.stride, self.padding)

    def init_weights(self):
        self.b = tf.Variable(tf.zeros(1,), trainable=True)
        if not self.param:
            self.w = normal(0, 0.1)(self.kernel_size)
        else:
            self.w = self.param(self.kernel_size)
        super().init_weights()

    def call(self, X):
        return conv2d(X, self.w, self.stride, self.padding) + self.b
