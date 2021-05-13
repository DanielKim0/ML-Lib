import tensorflow as tf

def pooling2d(X, size, padding, stride, mode="max"):
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

def pooling2d_multi(X, size, padding, stride, mode="max"):
    # assume channel-last syntax
    res = []
    for x in tf.split(X, X.shape[3], axis=3):
        res.append(x, size, padding, stride, mode)
    return tf.concat(res, 3)
