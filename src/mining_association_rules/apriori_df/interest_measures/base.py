from abc import ABCMeta, abstractmethod


class Measure(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.history: dict[frozenset[str], float] = {}
        self.calculations_count = 0
        self.calculations_time = 0.0

    @abstractmethod
    def calculate(self, *args, **kwargs):
        raise NotImplementedError
