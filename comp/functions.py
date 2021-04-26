import tensorflow as tf

def softmax(o):
    o_exp = tf.math.exp(o)
    sums = tf.math.reduce_sum(o_exp, 1, keepdims=True)
    return o_exp / sums

def accuracy(true, pred):
    maxes = tf.math.argmax(pred, axis=1)
    cmp = (tf.cast(maxes, true.dtype) == true)
    return float(tf.reduce_sum(tf.cast(cmp, true.dtype))) / len(pred)
