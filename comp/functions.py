import tensorflow as tf


def split_func(X, func, *args):
    inp = tf.split(X, X.shape[0], axis=0)
    args = list(args)
    res = [func(*[tf.squeeze(x, axis=0)] + list(args)) for x in inp]
    return tf.stack(res, 0)

