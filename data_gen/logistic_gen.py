from .base_gen import BaseGen
import tensorflow as tf

def softmax(o):
    o_exp = tf.math.exp(o)
    sums = tf.math.reduce_sum(o_exp, 1, keepdims=True)
    return o_exp / sums

class LogisticGen(BaseGen):
    def __init__(self):
        super().__init__()

    def create_batch(self, w, b, stddev, size=100):
        w = tf.cast(tf.constant(w), tf.float32)
        X = tf.zeros((size, w.shape[0]))
        X += tf.random.normal(shape=X.shape)
        y = softmax(tf.matmul(X, w) + b)
        y = tf.math.argmax(y, axis=1)
        return X, y
