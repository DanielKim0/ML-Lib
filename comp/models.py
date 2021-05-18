import tensorflow as tf

def linreg(X, w, b):
    return tf.matmul(X, w) + b

def logreg(X, w, b):
    return softmax(tf.matmul(X, w) + b)

def softmax(o):
    o_exp = tf.math.exp(o)
    sums = tf.math.reduce_sum(o_exp, 1, keepdims=True)
    return o_exp / sums
