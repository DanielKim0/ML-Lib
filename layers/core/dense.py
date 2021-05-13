import tensorflow as tf
from .core import CoreLayer
from comp.param_init import *

class DenseLayer(CoreLayer):
    def __init__(self, nodes, act=None, reg=None, param=None):
        super().__init__(act, reg, param)
        self.nodes = nodes
        self.w_mean = w_mean
        self.w_stddev = w_stddev

    def set_dims(self, inp):
        self.inp = inp
        self.out = self.nodes

    def init_weights(self):
        self.b = tf.Variable(tf.zeros(self.out[0]), trainable=True)
        if not self.param:
            self.w = normal(0, 0.1)((self.inp[0], self.out[0]))
        else:
            self.w = self.param((self.inp[0], self.out[0]))

    def call(self, X):
        return tf.matmul(X, self.w) + self.b
