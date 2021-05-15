import math
import tensorflow as tf

def pad_size(X, size, stride, padding):
    # how much should the pad_input function pad?
    # inputs shapes.
    if padding in ["none", "valid"]:
        return [[0, 0], [0, 0]]
    elif padding == "same":
        out = [math.ceil(X[i]/stride[i]) for i in range(len(stride))]
        total_pad = [(out[i]-1)*stride[i] + size[i] - X[i] for i in range(len(out))]
        return [[math.floor(total_pad[i]/2), math.ceil(total_pad[i]/2)] for i in range(len((total_pad)))]
    elif padding == "full":
        return # do this later!

def pad_input(X, size, stride, padding):
    pad_dims = pad_size(X.shape, size.shape, stride, padding)
    return tf.pad(X, pad_dims)
