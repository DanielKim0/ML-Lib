import tensorflow as tf
import numpy as np

from .base_model import BaseModel
from utils.compress import *


class LinearModel(BaseModel):
    def __init__(self):
        super().__init__()

    def __str__(self):
        s = "Linear Non-TF Model\n"
        if self.model_fit:
            s += "Currently fit\n"
            s += f"w.shape: {self.w.shape[0]}\n"
            s += f"intercept: {self.intercept}\n"
        else:
            s += "Currently not fit\n"
        return s

    def __repr__(self):
        if self.model_fit:
            s = f"LinearModel(model_fit={False})"
        else:
            s = f"LinearModel(model_fit={True}, w={self.w}, intercept={self.intercept})"
        return s

    def save(self, path):
        super().save()
        data = {
            "intercept": self.intercept
        }
        arrays = {
            "w": self.w
        }
        compress_files(path, data, arrays)

    def load(self, path):
        data, arrays = uncompress_files(path)
        self.intercept = data["intercept"]
        self.w = arrays["w"]

    def validate_fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("")
        if y.shape[1] != 1:
            raise ValueError("")

    def validate_predict(self, X):
        if X.shape[1] != self.w.shape[0]:
            raise ValueError("")

    def fit(self, X, y, intercept=False):
        X, y, self.intercept = X, y, intercept
        X = tf.cast(tf.constant(X), tf.float32)
        y = tf.cast(tf.constant(y), tf.float32)

        if self.intercept:
            X = self.add_intercept(X)
        self.validate_fit(X, y)

        XtX = tf.matmul(tf.transpose(X), X)
        XtX_inv = tf.linalg.inv(XtX)
        XtX_inv_Xt = tf.matmul(XtX_inv, tf.transpose(X))
        self.w = tf.matmul(XtX_inv_Xt, y)
        super().fit()

    def predict(self, X):
        if self.intercept:
            X = self.add_intercept(X)
        self.validate_predict(X)
        return tf.matmul(X, self.w)

    def add_intercept(self, X):
        intercept = tf.reshape(tf.ones(X.shape[0], dtype=tf.float32), [X.shape[0], 1])
        return tf.concat([X, intercept], axis=1)
