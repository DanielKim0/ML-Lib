import tensorflow as tf
from .core import CoreLayer
from comp.param_init import *

class DenseLayer(CoreLayer):
    def __init__(self, nodes, act=None, reg=None, param=None):
        super().__init__()
        self.nodes = nodes
        self.act = act
        self.reg = reg
        self.param = param
        self.w_mean = w_mean
        self.w_stddev = w_stddev

    def init_weights(self):
        self.b = tf.Variable(tf.zeros(self.out), trainable=True)
        if not self.param:
            self.w = normal(0, 0.1)((self.inp, self.out))
        else:
            self.w = self.param((self.inp, self.out))

    def op(self, X):
        if not self.act:
            return tf.matmul(X, self.w) + self.b
        else:
            return self.act(tf.matmul(X, self.w) + self.b)

    def loss(self):
        if not reg:
            return 0
        else:
            return reg(self.w)
