from perceptron import Perceptron
import numpy as np
import typing as tp


class Layer:
    def __init__(self):
        self.perceptrons: tp.List[Perceptron] = []
        pass

    def get_sum_future_grad(self, index: int) -> float:
        pass


class Layer:
    def __init__(self, num_in_features: int, num_out_features: int, is_hidden_layer: bool, initial_weights: tp.Optional[tp.List[np.ndarray]] = None,
                 initial_biases: tp.Optional[tp.List[float]] = None, use_biases: bool = True):
        """

        :param num_in_features: Number of inputs or nodes in previous layer.
        :param num_out_features: Number of outputs or nodes in next layer.
        :param is_hidden_layer: True if not output layer. False otherwise.
        :param initial_weights: Optional. Weights per neurons in enumeration.
        :param initial_biases: Optional. Biases per neurons in enumeration.
        :param use_biases: Optional. True by default. If True, neurons will use biases in their activity functions and will update the bias.
        """
        self.perceptrons: tp.List[Perceptron] = [Perceptron(num_in_features, is_hidden_node=is_hidden_layer, node_index=i,
                                                            initial_weights=initial_weights[i] if initial_weights is not None else None,
                                                            initial_bias=initial_biases[i] if initial_biases is not None else None,
                                                            use_bias=use_biases) for i in range(num_out_features)]
        self.num_in_features: int = num_in_features
        self.num_out_features: int = num_out_features
        self.is_hidden_layer: bool = is_hidden_layer

    def __str__(self) -> str:
        return f'Layer(in={self.num_in_features}, out={self.num_out_features}, is_hidden={self.is_hidden_layer}, use_biases={all(p.use_bias for p in self.perceptrons)})'

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.num_in_features:
            raise ValueError('I think youre not doing the forward pass correctly. Not sure though tbh')
        output: np.ndarray = np.array([p.get_output(x) for p in self.perceptrons])
        return output

    def get_sum_future_grad(self, index: int) -> float:
        # The calculation of this is correct.
        sum_future_grad: float = np.sum([p.get_sum_future_grad(index) for p in self.perceptrons])
        return sum_future_grad

    def calculate_updates(self, x: np.ndarray, lr: float, future_layer: tp.Optional[Layer] = None, desired_out: tp.Optional[np.ndarray] = None):
        """

        :param x: the output from the previous layer or the input to the network.
        :param lr:
        :param future_layer:
        :param desired_out:
        :return:
        """
        if self.is_hidden_layer:
            for p in self.perceptrons:
                p.calculate_updates(x, lr=lr, sum_future_grad=future_layer.get_sum_future_grad(p.node_index))
        else:
            for p in self.perceptrons:
                p.calculate_updates(x, lr=lr, desired_out=desired_out[p.node_index])

    def update_weights(self, x: np.ndarray, lr: float, desired_out: tp.Optional[np.ndarray] = None, future_layer: tp.Optional[Layer] = None,
                       update_weights: bool = True, update_biases: bool = True):
        self.calculate_updates(x, lr=lr, desired_out=desired_out, future_layer=future_layer)

        if self.is_hidden_layer:
            for p in self.perceptrons:
                p.update_perceptron(x, lr=lr, sum_future_grad=future_layer.get_sum_future_grad(p.node_index), update_weights=update_weights,
                                    update_biases=update_biases)
        else:
            for p in self.perceptrons:
                p.update_perceptron(x, lr=lr, desired_out=desired_out[p.node_index], update_weights=update_weights, update_biases=update_biases)
