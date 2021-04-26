from .metric import BaseMetric
import tensorflow as tf

class MSE(BaseMetric):
    def __str__(self):
        return "Mean Squared Error Metric"

    def __repr__(self):
        return "MSE()"

    def compare(self, true, pred):
        return tf.reduce_sum((true - pred)**2 / 2)
