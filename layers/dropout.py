import tensorflow as tf
from base import Layer


class DropoutLayer(Layer):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def __str__(self):
        return f"Dropout Layer: dropout={self.droupout}"

    def __repr__(self):
        return f"DropoutLayer(dropout={self.dropout})"

    def validate(self):
        if self.dropout <= 0 or self.dropout >= 1:
            return ValueError("")

    def op(self, X):
        mask = tf.random.uniform(shape=tf.shape(X), minval=0, maxval=1) < (1.0 - dropout)
        return tf.cast(maxk, dtype=tf.float32) * X / (1.0-dropout)
