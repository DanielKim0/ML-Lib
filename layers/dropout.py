import tensorflow as tf
from base import Layer

class Dropout(Layer):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def op(self, X):
        mask = tf.random.uniform(shape=tf.shape(X), minval=0, maxval=1) < (1.0 - dropout)
        return tf.cast(maxk, dtype=tf.float32) * X / (1.0-dropout)
