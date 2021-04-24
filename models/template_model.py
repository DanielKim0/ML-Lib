from .base_model import BaseModel

class TemplateModel(BaseModel):
    def __init__(self):
        super().__init__()

    def save(self):
        super().save()

    def load(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def validate_fit(self, **kwargs):
        pass

    def validate_predict(self, **kwargs):
        pass

    def fit(self, **kwargs):
        super().fit()
        pass

    def predict(self, **kwargs):
        pass

