import tensorflow as tf

def conv2d(X, K, padding, stride):
    h, w = K.shape
    if padding == "same":
        X = tf.pad(X, [[h-1, h-1], [w-1, w-1]])
    return conv2d_calc(X, k, padding)

def conv2d_calc(X, K, padding):
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0]-h+1, X.shape[1]-w+1)))
    for i in range(0, Y.shape[0], padding[0]):
        for j in range(0, Y.shape[1], padding[1]):
            Y[i,j].assign(tf.reduce_sum(X[i:i+h,j:j+w] * K))
    return Y
