import tensorflow as tf
from .metric import BaseMetric

class CrossEntropy(BaseMetric):
    def __str__(self):
        return "Cross-Entropy Metric"

    def __repr__(self):
        return "CrossEntropy()"

    def compare(self, true, pred):
        return -tf.math.log(tf.boolean_mask(pred, tf.one_hot(true, depth=pred.shape[-1])))
