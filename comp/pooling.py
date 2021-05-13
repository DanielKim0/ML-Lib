import tensorflow as tf

def pooling2d(X, size, stride, padding, mode="max"):
    if len(X.shape) == 4:
        return split_func(X, pooling2d_multi_calc, size, stride, padding, mode)
    else:
        return split_func(X, pooling2d_calc, size, stride, padding, mode)

def pooling2d_calc(X, size, stride, padding, mode):
    h, w = size
    if padding == "same":
        X = tf.pad(X, [[h-1, h-1], [w-1, w-1]])

    Y = tf.Variable(tf.zeros((X.shape[0]-h+1, X.shape[1]-w+1)))
    for i in range(0, Y.shape[0], stride[0]):
        for j in range(0, Y.shape[1], stride[1]):
            if mode == "max":
                tf.reduce_max(X[i:i+h, j:j+w])
            elif mode == "avg":
                tf.reduce_mean(X[i:i+h, j:j+w])
    return Y

def pooling2d_multi_calc(X, size, stride, padding, mode):
    # assume channel-last syntax
    res = []
    for x in tf.split(X, X.shape[2], axis=2):
        res.append(tf.squeeze(x, 2), size, stride, padding, mode)
    return tf.concat(res, 2)
