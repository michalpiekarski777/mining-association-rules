from abc import ABCMeta
from abc import abstractmethod


class Measure(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.calculations_count = 0
        self.calculations_time = 0.0

    @abstractmethod
    def calculate(self, *args, **kwargs):
        raise NotImplementedError
