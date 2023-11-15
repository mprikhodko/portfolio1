from layer import Layer
from perceptron import Perceptron
import typing as tp
import numpy as np
from copy import copy


class Network:
    def __init__(self):
        self.layers: tp.List[Layer] = []
        self.all_outputs: tp.List[np.ndarray] = []

    def __str__(self) -> str:
        out_lines: tp.List[str] = [f'({i}) :: {str(layer)}' for i, layer in enumerate(self.layers)]
        return '[\n\t' + '\n\t'.join(out_lines) + '\n]'

    def __repr__(self):
        return str(self)

    def feedforward(self, x: np.ndarray) -> np.ndarray:
        curr_data: np.ndarray = x.copy()
        all_outputs: tp.List[np.ndarray] = [curr_data]
        for layer in self.layers:
            curr_data: np.ndarray = layer.forward(curr_data)
            all_outputs.append(curr_data)
        self.all_outputs = all_outputs

        return curr_data

    @staticmethod
    def _reverse(l: tp.Iterable) -> tp.List:
        return [t for t in l][::-1]

    def get_total_error(self, desired_out: np.ndarray) -> float:
        return np.sum([p.get_big_error(self.all_outputs[-1], desired_out) for p in self.layers[-1].perceptrons])

    def calculate_updates(self, learning_rate: float, desired_out: np.ndarray):
        for i, layer in Network._reverse(enumerate(self.layers)):
            future_layer: Layer = self.layers[i + 1] if layer.is_hidden_layer else None
            true_output: tp.Optional[float] = desired_out if not layer.is_hidden_layer else None
            layer.calculate_updates(self.all_outputs[i], lr=learning_rate, future_layer=future_layer, desired_out=true_output)

    def backprop(self, learning_rate: float, desired_output: np.ndarray):
        self.calculate_updates(learning_rate, desired_output)

        for i, layer in enumerate(self.layers):
            future_layer: Layer = self.layers[i + 1] if layer.is_hidden_layer else None
            true_output: tp.Optional[float] = desired_output if not layer.is_hidden_layer else None
            layer.update_weights(self.all_outputs[i], lr=learning_rate, future_layer=future_layer, desired_out=true_output)

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        return self
