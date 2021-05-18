import tensorflow as tf
from comp.models import softmax
from .metric import BaseMetric


class CrossEntropy(BaseMetric):
    def __init__(self, default_logits=False):
        self.default_logits = default_logits

    def __str__(self):
        return "Cross-Entropy Metric"

    def __repr__(self):
        return "CrossEntropy()"

    def compare(self, true, pred, from_logits=None):
        if not from_logits:
            from_logits = self.default_logits

        if from_logits:
            pred = softmax(pred)
        return -tf.math.log(tf.boolean_mask(pred, tf.one_hot(true, depth=pred.shape[-1])))
