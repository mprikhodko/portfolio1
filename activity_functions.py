import typing as tp
import numpy as np
from abc import ABC, abstractmethod


class Activity(ABC):
    def __init__(self):
        self.last_inputs: np.ndarray = np.empty((0,))

    @abstractmethod
    def activity(self, x: np.ndarray) -> float:
        self.last_inputs = x.copy()
        pass

    @abstractmethod
    def derivative(self, name: str) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, name: str, value: tp.Any) -> tp.NoReturn:
        pass


class LinearActivity(Activity):
    def __init__(self, initial_weights: np.ndarray, initial_bias: float = 0, use_bias: bool = True):
        super(LinearActivity, self).__init__()
        self.weights: np.ndarray = initial_weights
        self.bias: float = initial_bias
        self.use_bias: bool = use_bias

    def activity(self, x: np.ndarray) -> float:
        super(LinearActivity, self).activity(x)
        if self.use_bias:
            return self.weights * x + self.bias
        else:
            return self.weights * x

    def derivative(self, name: str) -> np.ndarray:
        if name == 'bias':
            if not self.use_bias:
                raise ValueError('Cannot update bias if use_bias = False.')
            return np.ones((1, ))
        elif name == 'weights':
            return self.last_inputs

    def update(self, name: str, value: tp.Any) -> tp.NoReturn:
        if name == 'weights':
            if not isinstance(value, np.ndarray):
                raise ValueError(f'Expected a NumPy array. Got {type(value)}')
            self.weights = value
        elif name == 'bias':
            if not self.use_bias:
                raise ValueError('Cannot update bias if use_bias = False')
            elif not isinstance(value, type(self.bias)):
                raise ValueError(f'Cannot update bias due to type mismatch. Got {type(value)} when expected {type(self.bias)}')
            self.bias = value


class RadialBasisFunction(Activity):
    def __init__(self, initial_weights: np.ndarray):
        super(RadialBasisFunction, self).__init__()
        self.weights: np.ndarray = initial_weights.copy()

    def activity(self, x: np.ndarray) -> float:
        super(RadialBasisFunction, self).activity(x)
        return np.sqrt(np.sum((self.weights - x) ** 2))

    def derivative(self, name: str) -> np.ndarray:
        pass

    def update(self, name: str, value: tp.Any) -> tp.NoReturn:
        pass