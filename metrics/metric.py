from abc import abstractmethod

class BaseMetric:
    @abstractmethod
    def compare(self, true, pred):
        pass

    def validate(self):
        pass
