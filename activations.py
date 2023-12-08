import typing as tp
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np


class Activation(ABC):
    def __init__(self):
        pass

    def activity(self, x: np.ndarray, weights: np.ndarray, bias: float) -> float:
        return np.dot(x, weights) + bias

    @abstractmethod
    def activation(self, activity: float) -> float:
        pass

    @abstractmethod
    def derivative(self, activation: float) -> float:
        pass


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def activation(self, activity: float) -> float:
        return 1 / (1 + np.exp(-activity))

    def derivative(self, activation: float) -> float:
        return activation * (1 - activation)


class ReLu(Activation):
    def __init__(self, slope: float = 1):
        super(ReLu, self).__init__()
        self.slope = slope

    def activation(self, activity: float) -> float:
        return np.max([0, activity * self.slope])

    def derivative(self, activation: float) -> float:
        if activation <= 0:
            return 0
        return self.slope


class LeakyReLu(ReLu):
    def __init__(self, pos_slope: float, neg_slope: float):
        super(LeakyReLu, self).__init__(pos_slope)
        self.leaky_slope: float = neg_slope

    def activation(self, activity: float) -> float:
        if activity > 0:
            return self.slope * activity
        else:
            return self.leaky_slope * activity

    def derivative(self, activation: float) -> float:
        if activation <= 0:
            return self.leaky_slope
        else:
            return self.slope

