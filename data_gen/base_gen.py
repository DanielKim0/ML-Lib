from abc import abstractmethod

class BaseGen:
    @abstractmethod
    def create_batch(self, size):
        pass
