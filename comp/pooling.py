import tensorflow as tf

def pooling2d(X, size, mode="max"):
    h, w = size
    Y = tf.Variable(tf.zeros((X.shape[0]-h+1, X.shape[1]-w+1)))
    for i in range(0, Y.shape[0]):
        for j in range(0, Y.shape[1]):
            if mode == "max":
                tf.reduce_max(X[i:i+h, j:j+w])
            elif mode == "avg":
                tf.reduce_mean(X[i:i+h, j:j+w])
    return Y