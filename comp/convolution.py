import tensorflow as tf
from .functions import split_func
from .padding import * 

# 4d input with dimensions: batch, h, w, c_in
# 4d kernel with dimensions: k_h, k_w, c_in, c_out
# output: batch, o_h, o_w, c_out

def get_conv_size(inp, ker, stride, padding):
    if padding in ["none", "valid"]:
        size = [math.ceil((inp[i]-ker[i]+1)/stride[i]) for i in range(len(stride))]
    elif padding == "same":
        size = [math.ceil(inp[i]/stride[i]) for i in range(len(stride))]
        
    if len(inp) == 3: # if input has channel input
        if len(ker) == 4: # convolution
            size.append(ker[3])
        else: # pooling
            size.append(inp[2])
    return size

def conv2d(X, K, stride, padding):
    if len(X.shape) == 4:
        return split_func(X, conv2d_multi_calc, K, stride, padding)
    else:
        return split_func(X, conv2d_calc, K, stride, padding)

def conv2d_calc(X, K, stride, padding):
    h, w = K.shape
    Y = tf.Variable(tf.zeros(get_conv_size(X.shape, K.shape, stride, padding)))
    X = pad_input(X, K, stride, padding)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(X[i*stride[0]:i*stride[0]+h,j*stride[1]:j*stride[1]+w] * K))
    return Y

def conv2d_multi_calc(X, kernel, stride, padding):
    channels = []
    # split across output channels
    for K in tf.split(kernel, kernel.shape[3], axis=3):
        K = tf.squeeze(K, axis=3)
        # split across input channels
        X_split = tf.split(X, X.shape[2], axis=2)
        K_split = tf.split(K, K.shape[2], axis=2)
        conv = [conv2d_calc(tf.squeeze(X_split[i], axis=2), tf.squeeze(K_split[i], axis=2), stride, padding) for i in range(len(X_split))]
        # stack the split results back to input channels, and sum across input channels
        conv = tf.stack(conv, axis=2)
        channels.append(tf.reduce_sum(conv, axis=2))
    # stack the split output channel results back to proper dimension (c_out)
    return tf.stack(channels, axis=2)
