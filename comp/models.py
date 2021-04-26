import tensorflow as tf
from .functions import *

def linreg(X, w, b):
    return tf.matmul(X, w) + b

def logreg(X, w, b):
    return softmax(tf.matmul(X, w) + b)
