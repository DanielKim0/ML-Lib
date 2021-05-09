import tensorflow as tf
from .metric import BaseMetric
from comp.functions import softmax

class CrossEntropy(BaseMetric):
    def __str__(self):
        return "Cross-Entropy Metric"

    def __repr__(self):
        return "CrossEntropy()"

    def compare(self, true, pred, from_logits=False):
        if from_logits:
            pred = softmax(pred)
        return -tf.math.log(tf.boolean_mask(pred, tf.one_hot(true, depth=pred.shape[-1])))
