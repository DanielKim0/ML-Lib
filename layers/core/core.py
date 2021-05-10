from abc import abstractmethod
import tensorflow as tf
from ..base import Layer


class CoreLayer(Layer):
    def __init__(self):
        super().__init__()
        self.w = None
        self.b = None
        self.weighted = True

    def apply_grad(self, grads):
        for param, grad in zip([self.w, self.b], grads):
            param.assign_sub(param, grad)

    def weights(self):
        return [self.w, self.b]

    @abstractmethod
    def init_weights(self):
        pass
