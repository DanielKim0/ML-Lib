import tensorflow as tf

def conv2d(X, K, padding, stride):
    h, w = K.shape
    if padding == "same":
        X = tf.pad(X, [[h-1, h-1], [w-1, w-1]])
    return conv2d_calc(X, k, stride)

def conv2d_calc(X, K, stride):
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0]-h+1, X.shape[1]-w+1)))
    for i in range(0, Y.shape[0], stride[0]):
        for j in range(0, Y.shape[1], stride[1]):
            Y[i,j].assign(tf.reduce_sum(X[i:i+h,j:j+w] * K))
    return Y

def conv2d_multi(X, kernel, padding, stride):
    channels = []
    for K in kernel:
        channels.append(tf.reduce_sum([conv2d(x, k, padding, stride) for x, k in zip(X, K)], axis=0))
    return tf.stack(channels, 0)

def conv2d_multi_1x1(X, K):
    conv_in, h, w = X.shape
    conv_out = K.shape[0]
    X = tf.reshape(X, (conv_in, h * w))
    K = tf.reshape(K, (conv_out, conv_in))
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (conv_out, h, w))
