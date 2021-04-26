import tensorflow as tf
from abc import abstractmethod
from .base_model import BaseModel
import random, os

class TFModel(BaseModel):
    def __init__(self, checkpoint_dir="checkpoints"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.global_step = 0
        self.curr_epoch = 0

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
