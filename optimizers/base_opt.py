from abc import abstractmethod


class BaseOpt:
    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def validate(self):
        pass
