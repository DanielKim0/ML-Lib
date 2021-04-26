from .metric import BaseMetric
import tensorflow as tf


class ClassAccuracy(BaseMetric):
    def compare(self, true, pred):
        maxes = tf.math.argmax(pred, axis=1)
        cmp = (tf.cast(maxes, true.dtype) == true)
        return float(tf.reduce_sum(tf.cast(cmp, true.dtype))) / len(pred)