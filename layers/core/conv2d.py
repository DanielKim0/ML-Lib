import tensorflow as tf
from comp.convolution import *
from comp.param_init import *

class Conv2DLayer(CoreLayer):
    def __init__(self, nodes, kernel_size, padding="valid", stride=(1,1), act=None, reg=None, param=None):
        super().__init__(act, reg, param)
        self.nodes = nodes
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def set_dims(self, inp, out):
        self.inp = inp
        self.out = get_conv_size(inp, self.kernel_size, self.stride, self.padding)

    def init_weights(self, kernel_size):
        self.b = tf.Variable(tf.zeros(1,), trainable=True)
        if not self.param:
            self.w = normal(0, 0.1)(kernel_size)
        else:
            self.w = self.param(kernel_size)

    def call(self, X):
        return conv2d(X, self.w, self.padding, self.stride) + self.b
