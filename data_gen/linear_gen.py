from .base_gen import BaseGen
import tensorflow as tf

class LinearGen(BaseGen):
    def __init__(self, w, b, stddev):
        super().__init__()
        self.w = tf.cast(tf.constant(w), tf.float32)
        self.b = tf.cast(tf.constant(b), tf.float32)
        self.stddev = stddev

    def create_batch(self, size=100):
        X = tf.zeros((size, self.w.shape[0]))
        X += tf.random.normal(shape=X.shape)
        y = tf.matmul(X, tf.reshape(self.w, (-1, 1))) + self.b
        y += tf.random.normal(shape=y.shape, stddev=self.stddev)
        y = tf.reshape(y, (-1, 1))
        return X, y
