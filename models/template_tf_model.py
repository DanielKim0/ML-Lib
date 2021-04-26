import tensorflow as tf

from .tf_model import TFModel


class TemplateTFModel(TFModel):
    def __init__(self):
        super().__init__()

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def save(self):
        super().save()

    def load(self):
        pass

    def validate_fit(self):
        pass

    def validate_predict(self):
        pass

    def build_model(self):
        pass

    def fit(self):
        super().fit()
        pass

    def train_epoch(self):
        pass

    def train_step(self):
        pass

    def predict(self):
        pass
