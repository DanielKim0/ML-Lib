from abc import abstractmethod

class BaseModel:
    def __init__(self):
        self.model_fit = False

    @abstractmethod
    def save(self):
        if not self.model_fit:
            raise ValueError("Cannot save model if it has not been fit!")

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def validate_fit(self, **kwargs):
        pass

    @abstractmethod
    def validate_predict(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        self.model_fit = True

    @abstractmethod
    def predict(self, **kwargs):
        pass
