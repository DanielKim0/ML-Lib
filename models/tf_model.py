import os
import random
import tensorflow as tf
from abc import abstractmethod
from .base_model import BaseModel


class TFModel(BaseModel):
    def __init__(self, checkpoint_dir="checkpoints"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0
        self.curr_epoch = 0

    def validate_model(self, stddev):
        if stddev <= 0:
            raise ValueError("")

    def validate_fit(self, X, y, batch_size, num_epochs, opt, loss):
        if X.shape[0] != y.shape[0]:
            raise ValueError("")
        if y.shape[1] != 1:
            raise ValueError("")
        if batch_size <= 0 or not isinstance(batch_size, int):
            raise ValueError("")
        if num_epochs <= 0 or not isinstance(num_epochs, int):
            raise ValueError("")
        opt.validate()
        loss.validate()

    def validate_predict(self, X):
        if X.shape[1] != self.w.shape[0]:
            raise ValueError("")

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    def fit(self, *args):
        super().fit()
        for cur_epoch in range(self.curr_epoch, self.num_epochs):
            self.curr_epoch += 1
            self.train_epoch(*args)

    def data_iter(self, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        # The examples are read at random, in no particular order
        random.shuffle(indices)
        for i in range(0, num_examples, self.batch_size):
            j = tf.constant(indices[i:min(i + self.batch_size, num_examples)])
            yield tf.gather(features, j), tf.gather(labels, j)
