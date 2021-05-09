import tensorflow as tf

def l1_penalty(w, lambd=1):
    return lambd * tf.reduce_sum(tf.abs(w))

def l2_penalty(w, lambd=1):
    return lambd * tf.reduce_sum(tf.pow(w, 2)) / 2
