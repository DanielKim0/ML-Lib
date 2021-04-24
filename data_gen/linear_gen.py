from .base_gen import BaseGen
import tensorflow as tf

class LinearGen(BaseGen):
    def __init__(self):
        super().__init__()

    def create_batch(self, w, b, stddev, size=100):
        w = tf.constant(w)
        X = tf.zeros((size, w.shape[0]))
        X += tf.random.normal(shape=X.shape)
        y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
        y += tf.random.normal(shape=y.shape, stddev=stddev)
        y = tf.reshape(y, (-1, 1))
        return X, y
