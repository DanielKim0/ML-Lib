import tensorflow as tf

def relu(x):
    return tf.maximum(x, 0)

def leaky_relu(x, alpha=0.01):
    return tf.maximum(x, alpha*x)

def prelu(x, alpha):
    return tf.maximum(x, 0) + alpha * tf.minimum(x, 0)

def sigmoid(x):
    return tf.math.divide(1, (1 + tf.math.exp(-x)))

def tanh(x):
    return tf.math.divide(1 - tf.math.exp(-2 * x), 1 + tf.math.exp(-2 * x))
    