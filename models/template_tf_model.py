from .tf_model import TFModel
import tensorflow as tf


class TemplateTFModel(TFModel):
    def __init__(self):
        super().__init__()

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

    def predict(self):
        pass
