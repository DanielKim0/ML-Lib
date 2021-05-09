import math
import tensorflow as tf
from .base_gen import BaseGen


class PolynomialGen(BaseGen):
    def __init__(self, feature_size, max_power, stddev=0.1, rescale=True):
        super().__init__()
        self.feature_size = feature_size
        self.max_power = max_power
        self.stddev = stddev
        self.rescale = rescale
        self.w = self.gen_w()
        self.b = self.gen_b()

    def __str__(self):
        s = "Polynomial Data Generator:\n"
        s += f"feature_size = {self.feature_size}\n"
        s += f"max_power = {self.max_power}"
        s += f"stddev = {self.stddev}\n"
        s += f"rescale = {self.rescale}\n"
        return s

    def __repr__(self):
        return f"PolynomialGen(feature_size={self.feature_size}, max_power={self.max_power}, stddev={self.stddev}, rescale={self.rescale})"

    def gen_w(self):
        return tf.random.normal(shape=(self.feature_size, self.max_power))

    def gen_b(self):
        return tf.random.normal(shape=(1,))

    def create_batch(self, size=100):
        X = tf.zeros((size, self.feature_size))
        X += tf.random.normal(shape=X.shape)
        
        splits = tf.split(self.w, num_or_size_splits=self.w.shape[1], axis=1)
        y = tf.zeros((size, 1))
        expo = 1

        for split in splits:
            split = tf.reshape(split, (-1, 1))
            curr_sum = tf.matmul(tf.math.exp(X, expo), split)
            if self.rescale:
                curr_sum /= math.factorial(expo)
            y += curr_sum
            expo += 1
        y += self.b
        y += tf.random.normal(shape=y.shape, stddev=self.stddev)
        return X, y

