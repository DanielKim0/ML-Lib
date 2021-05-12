from abc import abstractmethod
import tensorflow as tf
from ..base import Layer


class CoreLayer(Layer):
    def __init__(self, act, reg, param):
        super().__init__()
        self.w = None
        self.b = None
        self.weighted = True
        self.act = act
        self.reg = reg
        self.param = param

    def apply_grad(self, grads):
        for param, grad in zip([self.w, self.b], grads):
            param.assign_sub(param, grad)

    def weights(self):
        return [self.w, self.b]

    def op(self, X):
        if not self.act:
            return self.call(X)
        else:
            return self.act(self.call(X))

    def loss(self):
        if not reg:
            return 0
        else:
            return reg(self.w)

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def call(self, X):
        pass
