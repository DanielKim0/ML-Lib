from abc import abstractmethod
import tensorflow as tf


class Layer:
    def __init__(self):
        self.nodes = None
        self.weighted = False
        self.inp = 0
        self.out = 0

    def set_dims(self, inp):
        self.inp = inp
        self.out = inp

    def validate(self):
        pass

    @abstractmethod
    def op(self):
        pass
