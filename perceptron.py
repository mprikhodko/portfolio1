import numpy as np
import typing as tp
import activations
from activity_functions import *


class Perceptron:

    def __init__(self, num_inputs: int, is_hidden_node: bool, node_index: int = 0, initial_weights: tp.Optional[np.ndarray] = None,
                 initial_bias: tp.Optional[float] = None, use_bias: bool = True, activation: activations.Activation = None, activity: Activity = None):
        self.weights: np.ndarray = np.random.random(size=num_inputs) if initial_weights is None else initial_weights
        self.bias: float = np.random.random(None) if initial_bias is None else initial_bias
        self.node_index: int = node_index
        self.delta_value: tp.Optional[float] = None
        self.is_hidden_node: bool = is_hidden_node
        self.use_bias: bool = use_bias
        self.activation_function: activations.Activation = activation or activations.Sigmoid()
        self.activity_function: Activity = activity or LinearActivity(self.weights, self.bias, self.use_bias)

    def get_activation(self, x: float) -> float:
        # Sigmoid
        return self.activation_function.activation(x)

    def get_derivative_activation(self, activation: float) -> float:
        return self.activation_function.derivative(activation)

    def get_activity(self, x: np.ndarray) -> float:
        return self.activity_function.activity(x)
        # # For a node j
        # out: float = np.sum(self.weights * x)
        # if self.use_bias:
        #     return out + self.bias
        # return out

    def get_output(self, x: np.ndarray) -> float:
        return self.get_activation(self.get_activity(x))

    def get_big_error(self, our_out: float, desired_out: float) -> float:
        # Assumes error is half of Euclidean distance.
        return 0.5 * (self.get_small_error(our_out, desired_out)) ** 2

    def get_small_error(self, our_out: float, desired_out: float) -> float:
        return our_out - desired_out

    def _calculate_delta(self, our_out: float, adjuster_value: float):
        """

        :param our_out:
        :param adjuster_value: Either desired_out if self.is_hidden_layer == False, otherwise should be equal to the future layer's sum of gradients.
        :return:
        """
        if adjuster_value is None:
            error_msg: str = 'Make sure sum_future_grad is not None' if self.is_hidden_node else 'Make sure desired_out is not None'
            raise ValueError('Hey the future value should be correct. ' + error_msg)
        if self.is_hidden_node:
            self.delta_value: float = adjuster_value * self.get_derivative_activation(our_out)
        else:
            self.delta_value: float = -self.get_small_error(our_out, adjuster_value) * self.get_derivative_activation(our_out)

    def get_delta_weights(self, x: np.ndarray, lr: float, desired_out: tp.Optional[float] = None,
                          sum_future_grad: tp.Optional[float] = None) -> np.ndarray:
        our_out: float = self.get_output(x)
        if self.delta_value is None:
            self._calculate_delta(our_out, desired_out if not self.is_hidden_node else sum_future_grad)
        delta_Ej_over_wj: np.ndarray = self.delta_value * self.activity_function.derivative('weights')
        return lr * delta_Ej_over_wj

    def get_delta_bias(self, x: np.ndarray, lr: float, desired_out: tp.Optional[float] = None, sum_future_grad: tp.Optional[float] = None) -> float:
        our_out: float = self.get_output(x)
        if self.delta_value is None:
            self._calculate_delta(our_out, desired_out if not self.is_hidden_node else sum_future_grad)
        return lr * self.delta_value * self.activity_function.derivative('bias').item()

    def calculate_updates(self, x: np.ndarray, desired_out: tp.Optional[float] = None, sum_future_grad: tp.Optional[float] = None,
                          lr: float = 1e-3) -> tp.Tuple[np.ndarray, float]:
        delta_weights: np.ndarray = self.get_delta_weights(x, lr, desired_out=desired_out, sum_future_grad=sum_future_grad)
        delta_bias: float = self.get_delta_bias(x, lr, desired_out=desired_out, sum_future_grad=sum_future_grad) if self.use_bias else 0
        return delta_weights, delta_bias

    def update_perceptron(self, x: np.ndarray, desired_out: tp.Optional[float] = None, sum_future_grad: tp.Optional[float] = None,
                          lr: float = 1e-3, update_weights: bool = True, update_biases: bool = True) -> tp.NoReturn:
        delta_weights, delta_bias = self.calculate_updates(x, lr=lr, desired_out=desired_out, sum_future_grad=sum_future_grad)

        if update_weights:
            self.weights += delta_weights
        if self.use_bias and update_biases:
            self.bias += delta_bias

        self.activity_function.update('bias', self.bias)
        self.activity_function.update('weights', self.weights)

        self.delta_value = None

    def get_sum_future_grad(self, k: int) -> np.ndarray:
        return self.weights[k] * self.delta_value
