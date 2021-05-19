import tensorflow as tf
from comp.param_init import normal
from .core import CoreLayer


class DenseLayer(CoreLayer):
    def __init__(self, nodes, act=None, reg=None, param=None):
        super().__init__(act, reg, param)
        # w_mean and w_stddev are now in "param"
        self.nodes = nodes

    def __str__(self):
        s = "Dense Layer\n"
        if self.initialized:
            s += "Currently initialized\n"
            s += f"inp: {self.inp}\n"
            s += f"out: {self.out}\n"
            s += f"w.shape: {self.w.shape}\n"
        else:
            s += "Currently not initialized\n"
        s += f"nodes: {self.nodes}\n"
        s += f"act: {self.act}\n"
        s += f"reg: {self.reg}\n"
        s += f"param: {self.param}\n"
        return s

    def __repr__(self):
        if self.initialized:
            s = f"DenseLayer(initialized={False}, nodes={self.nodes}, act={self.act}, reg={self.reg}, param={self.param})"
        else:
            s = f"DenseLayer(initialized={True}, inp={self.inp}, out={self.out}, w.shape={self.w.shape}, nodes={self.nodes}, act={self.act}, reg={self.reg}, param={self.param})"
        return s

    def set_dims(self, inp):
        self.inp = inp
        self.out = [self.nodes]

    def init_weights(self):
        self.b = tf.Variable(tf.zeros(self.out[0]), trainable=True)
        if not self.param:
            self.w = normal(0, 0.1)((self.inp[0], self.out[0]))
        else:
            self.w = self.param((self.inp[0], self.out[0]))
        super().init_weights()

    def call(self, X):
        return tf.matmul(X, self.w) + self.b
