import tensorflow as tf
from .metric import BaseMetric

class CrossEntropy(BaseMetric):
    def compare(self, true, pred):
        return -tf.math.log(tf.boolean_mask(pred, tf.one_hot(true, depth=pred.shape[-1])))
