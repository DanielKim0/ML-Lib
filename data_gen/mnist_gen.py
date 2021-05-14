import tensorflow as tf
from .base_gen import BaseGen


class MNISTGen(BaseGen):
    def __init__(self):
        super().__init__()

    def __str__(self):
        s = "MNIST Data Generator:\n"
        return s

    def __repr__(self):
        return f"MNISTGen()"

    def create_batch(self, channel=True):
        if not channel:
            return tf.keras.datasets.mnist.load_data()
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = tf.expand_dims(X_train, axis=-1)
        X_test = tf.expand_dims(X_test, axis=-1)
        return (X_train, y_train), (X_test, y_test)