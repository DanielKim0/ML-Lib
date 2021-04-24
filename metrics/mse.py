from .metric import BaseMetric
import tensorflow as tf

class MSE(BaseMetric): 
    def compare(self, true, pred):
        return tf.reduce_sum((true - pred)**2 / 2)