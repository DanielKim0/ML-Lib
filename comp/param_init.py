import math
import tensorflow as tf

def normal(mean, std):
    def init(size):
        return tf.Variable(tf.random.normal(size, mean=mean, stddev=std), trainable=True)
    return init

def xavier(size):
    inp, out = size
    return tf.Variable(tf.random.uniform(size, -math.sqrt(6/(inp + out)), math.sqrt(6/(inp + out))))
