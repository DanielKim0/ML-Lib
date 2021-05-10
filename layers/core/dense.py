import tensorflow as tf
from .core import CoreLayer

class DenseLayer(CoreLayer):
    def __init__(self, nodes, act=None, w_mean=0, w_stddev=0.1):
        super().__init__()
        self.nodes = nodes
        self.act = act
        self.w_mean = w_mean
        self.w_stddev = w_stddev

    def init_weights(self):
        self.b = tf.Variable(tf.zeros(self.out), trainable=True)
        self.w = tf.Variable(tf.random.normal((self.inp, self.out), mean=self.w_mean, stddev=self.w_stddev), trainable=True)

    def op(self, X):
        if not self.act:
            return tf.matmul(X, self.w) + self.b
        else:
            return self.act(tf.matmul(X, self.w) + self.b)
