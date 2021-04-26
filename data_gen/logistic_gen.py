import tensorflow as tf
from .base_gen import BaseGen


def softmax(o):
    o_exp = tf.math.exp(o)
    sums = tf.math.reduce_sum(o_exp, 1, keepdims=True)
    return o_exp / sums

class LogisticGen(BaseGen):
    def __init__(self, w, b, stddev):
        super().__init__()
        self.w = tf.cast(tf.constant(w), tf.float32)
        self.b = tf.cast(tf.constant(b), tf.float32)
        self.stddev = stddev

    def create_batch(self, size=100):
        X = tf.zeros((size, self.w.shape[0]))
        X += tf.random.normal(shape=X.shape)
        y = tf.matmul(X, self.w) + self.b
        y = softmax(y + tf.random.normal(shape=(X.shape[0], self.w.shape[1]), stddev=self.stddev))
        y = tf.math.argmax(y, axis=1)
        return X, y
