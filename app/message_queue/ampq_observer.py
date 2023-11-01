from abc import ABCMeta, abstractmethod


class AmqpObserver(metaclass=ABCMeta):
    @abstractmethod
    def update(self, data):
        pass
