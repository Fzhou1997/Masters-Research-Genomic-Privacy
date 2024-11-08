from typing import Sequence, Type

import torch
import torch.nn as nn
from torch import Tensor

from .ModelAttackerLSTM import ModelAttackerLSTM
from utils_torch.modules import MultiLayerLinear, hx_type

class ModelAttackerLSTMLinear(ModelAttackerLSTM):
    """
    A PyTorch module for an LSTM-Linear-based attacker model.

    Attributes:
        _lstm_linear_dropout_p (float): The dropout probability between the LSTM and linear layers.
        _lstm_linear_dropout_first (bool): If True, apply dropout before batch normalization.
        _lstm_linear_batch_norm (bool): If True, apply batch normalization between the LSTM and linear layers.
        _lstm_linear_batch_norm_momentum (float): The momentum for batch normalization.
        lstm_linear_dropout_module (nn.Dropout | nn.Identity): The dropout module between the LSTM and linear layers.
        lstm_linear_batch_norm_module (nn.BatchNorm1d | nn.Identity): The batch normalization module between the LSTM and linear layers.
        linear_modules (MultiLayerLinear): The linear layers used in the model.
    """

    _lstm_linear_dropout_p: float
    _lstm_linear_dropout_first: bool

    _lstm_linear_batch_norm: bool
    _lstm_linear_batch_norm_momentum: float

    lstm_linear_dropout_module: nn.Dropout | nn.Identity
    lstm_linear_batch_norm_module: nn.BatchNorm1d | nn.Identity
    linear_modules: MultiLayerLinear

    def __init__(self,
                 lstm_num_layers: int,
                 lstm_input_size: int,
                 lstm_hidden_size: int | Sequence[int],
                 linear_num_layers: int,
                 linear_num_features: int | Sequence[int],
                 lstm_proj_size: int | Sequence[int] = 0,
                 lstm_bidirectional: bool | Sequence[bool] = False,
                 lstm_dropout_p: float | Sequence[float] = 0.5,
                 lstm_dropout_first: bool | Sequence[bool] = True,
                 lstm_layer_norm: bool | Sequence[bool] = True,
                 lstm_linear_dropout_p: float = 0.25,
                 lstm_linear_dropout_first: bool = True,
                 lstm_linear_batch_norm: bool = True,
                 lstm_linear_batch_norm_momentum: float = 0.1,
                 linear_activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 linear_activation_kwargs: dict[str, any] | Sequence[dict[str, any]] = None,
                 linear_dropout_p: float | Sequence[float] = 0.5,
                 linear_dropout_first: bool | Sequence[bool] = True,
                 linear_batch_norm: bool | Sequence[bool] = True,
                 linear_batch_norm_momentum: float | Sequence[float] = 0.1,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes the ModelAttackerLSTMLinear class.

        Args:
            lstm_input_size (int): The number of expected features in the input.
            lstm_hidden_size (int | Sequence[int]): The number of features in the hidden state.
            lstm_num_layers (int, optional): Number of recurrent layers. Default is 1.
            lstm_proj_size (int | Sequence[int], optional): If > 0, will use LSTM with projections of corresponding size. Default is 0.
            lstm_bidirectional (bool | Sequence[bool], optional): If True, becomes a bidirectional LSTM. Default is False.
            lstm_dropout_p (float | Sequence[float], optional): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer. Default is 0.5.
            lstm_dropout_first (bool | Sequence[bool], optional): If True, applies dropout before batch normalization. Default is True.
            lstm_layer_norm (bool | Sequence[bool], optional): If True, applies layer normalization. Default is True.
            lstm_linear_dropout_p (int, optional): Dropout probability between the lstm and linear layers. Default is 0.25.
            lstm_linear_dropout_first (bool, optional): If True, applies dropout before batch normalization between the lstm and linear layers. Default is True.
            lstm_linear_batch_norm (bool, optional): If True, applies batch normalization between the lstm and linear layers. Default is True.
            lstm_linear_batch_norm_momentum (float, optional): Momentum for batch normalization between the lstm and linear layers. Default is 0.1.
            linear_num_layers (int, optional): Number of linear layers. Default is 1.
            linear_num_features (int | Sequence[int], optional): Number of features in the linear layers. Default is 8.
            linear_activation (Type[nn.Module] | Sequence[Type[nn.Module]], optional): Activation function for the linear layers. Default is nn.ReLU.
            linear_activation_kwargs (dict[str, any] | Sequence[dict[str, any]], optional): Keyword arguments for the activation function. Default is None.
            linear_dropout_p (float | Sequence[float], optional): Dropout probability for the linear layers. Default is 0.5.
            linear_dropout_first (bool | Sequence[bool], optional): If True, applies dropout before batch normalization in linear layers. Default is True.
            linear_batch_norm (bool | Sequence[bool], optional): If True, applies batch normalization in linear layers. Default is True.
            linear_batch_norm_momentum (float | Sequence[float], optional): Momentum for batch normalization in linear layers. Default is 0.1.
            device (torch.device, optional): The device on which to place the model. Default is None.
            dtype (torch.dtype, optional): The data type of the model parameters. Default is None.
        """
        super(ModelAttackerLSTMLinear, self).__init__(lstm_num_layers=lstm_num_layers,
                                                      lstm_input_size=lstm_input_size,
                                                      lstm_hidden_size=lstm_hidden_size,
                                                      lstm_proj_size=lstm_proj_size,
                                                      lstm_bidirectional=lstm_bidirectional,
                                                      lstm_dropout_p=lstm_dropout_p,
                                                      lstm_dropout_first=lstm_dropout_first,
                                                      lstm_layer_norm=lstm_layer_norm,
                                                      device=device,
                                                      dtype=dtype)

        assert self.lstm_output_size_out == linear_num_features[0], 'The output size of the LSTM must match the input size of the linear layers.'

        self._lstm_linear_dropout_p = lstm_linear_dropout_p
        self._lstm_linear_dropout_first = lstm_linear_dropout_first

        self._lstm_linear_batch_norm = lstm_linear_batch_norm
        self._lstm_linear_batch_norm_momentum = lstm_linear_batch_norm_momentum

        if lstm_linear_dropout_p > 0:
            self.lstm_linear_dropout_module = nn.Dropout(p=lstm_linear_dropout_p)
        else:
            self.lstm_linear_dropout_module = nn.Identity()

        if lstm_linear_batch_norm:
            self.lstm_linear_batch_norm_module = nn.BatchNorm1d(num_features=self.lstm_output_size_out,
                                                                momentum=lstm_linear_batch_norm_momentum,
                                                                device=device,
                                                                dtype=dtype)
        else:
            self.lstm_linear_batch_norm_module = nn.Identity()

        self.linear_modules = MultiLayerLinear(num_layers=linear_num_layers,
                                               num_features=linear_num_features,
                                               activation=linear_activation,
                                               activation_kwargs=linear_activation_kwargs,
                                               dropout_p=linear_dropout_p,
                                               dropout_first=linear_dropout_first,
                                               batch_norm=linear_batch_norm,
                                               batch_norm_momentum=linear_batch_norm_momentum,
                                               device=device,
                                               dtype=dtype)

    def forward(self,
                x: Tensor,
                hx: hx_type = None) -> tuple[Tensor, hx_type]:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.
            hx (hx_type, optional): Initial hidden state. Default is None.

        Returns:
            tuple[Tensor, hx_type]: Output tensor and hidden state.
        """
        _, (hy, cy) = self.lstm_modules.forward(x, hx)
        hy_last = hy[-1]
        if self.lstm_bidirectional[-1]:
            hy_last = torch.cat((hy_last[-2], hy_last[-1]), dim=1)
        else:
            hy_last = hy_last[-1]
        y = hy_last
        if self.lstm_linear_dropout_first:
            y = self.lstm_linear_dropout_module(y)
            y = self.lstm_linear_batch_norm_module(y)
        else:
            y = self.lstm_linear_batch_norm_module(y)
            y = self.lstm_linear_dropout_module(y)
        y = self.linear_modules(y).squeeze()
        return y, (hy, cy)

    def predict(self, logits: Tensor) -> Tensor:
        """
        Applies a sigmoid function to the logits to get predictions.

        Args:
            logits (Tensor): Logits tensor.

        Returns:
            Tensor: Predictions tensor.
        """
        return torch.sigmoid(logits)

    def classify(self, predicted: Tensor) -> Tensor:
        """
        Rounds the predictions to get binary classification.

        Args:
            predicted (Tensor): Predictions tensor.

        Returns:
            Tensor: Binary classification tensor.
        """
        return torch.round(predicted)

    @property
    def lstm_linear_dropout_p(self) -> float:
        """
        Returns the dropout probability for the dropout layer between the LSTM and linear layers.

        Returns:
            float: Dropout probability.
        """
        return self._lstm_linear_dropout_p

    @property
    def lstm_linear_dropout_first(self) -> bool:
        """
        Returns whether dropout is applied before batch normalization between the LSTM and linear layers.

        Returns:
            bool: True if dropout is applied first, False otherwise.
        """
        return self._lstm_linear_dropout_first

    @property
    def lstm_linear_batch_norm(self) -> bool:
        """
        Returns whether batch normalization is applied between the LSTM and linear layers.

        Returns:
            bool: True if batch normalization is applied, False otherwise.
        """
        return self._lstm_linear_batch_norm

    @property
    def lstm_linear_batch_norm_momentum(self) -> float:
        """
        Returns the momentum for batch normalization between the LSTM and linear layers.

        Returns:
            float: Momentum for batch normalization.
        """
        return self._lstm_linear_batch_norm_momentum

    @property
    def linear_num_layers(self) -> int:
        """
        Returns the number of linear layers.

        Returns:
            int: Number of linear layers.
        """
        return self.linear_modules.linear_num_layers

    @property
    def linear_num_features_in(self) -> int:
        """
        Returns the number of input features for the linear layers.

        Returns:
            int: Number of input features.
        """
        return self.linear_modules.linear_num_features_in

    @property
    def linear_num_features_out(self) -> int:
        """
        Returns the number of output features for the linear layers.

        Returns:
            int: Number of output features.
        """
        return self.linear_modules.linear_num_features_out

    @property
    def linear_num_features(self) -> tuple[int, ...]:
        """
        Returns the number of features for each linear layer.

        Returns:
            tuple[int, ...]: Number of features for each linear layer.
        """
        return self.linear_modules.linear_num_features

    @property
    def linear_activation(self) -> tuple[Type[nn.Module], ...]:
        """
        Returns the activation functions for the linear layers.

        Returns:
            tuple[Type[nn.Module], ...]: Activation functions for the linear layers.
        """
        return self.linear_modules.linear_activation

    @property
    def linear_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        """
        Returns the keyword arguments for the activation functions in the linear layers.

        Returns:
            tuple[dict[str, any], ...]: Keyword arguments for the activation functions.
        """
        return self.linear_modules.linear_activation_kwargs

    @property
    def linear_dropout_p(self) -> tuple[float, ...]:
        """
        Returns the dropout probabilities for the linear layers.

        Returns:
            tuple[float, ...]: Dropout probabilities.
        """
        return self.linear_modules.linear_dropout_p

    @property
    def linear_dropout_first(self) -> tuple[bool, ...]:
        """
        Returns whether dropout is applied before batch normalization in the linear layers.

        Returns:
            tuple[bool, ...]: True if dropout is applied first, False otherwise.
        """
        return self.linear_modules.linear_dropout_first

    @property
    def linear_batch_norm(self) -> tuple[bool, ...]:
        """
        Returns whether batch normalization is applied in the linear layers.

        Returns:
            tuple[bool, ...]: True if batch normalization is applied, False otherwise.
        """
        return self.linear_modules.linear_batch_norm

    @property
    def linear_batch_norm_momentum(self) -> tuple[float, ...]:
        """
        Returns the momentum for batch normalization in the linear layers.

        Returns:
            tuple[float, ...]: Momentum for batch normalization.
        """
        return self.linear_modules.linear_batch_norm_momentum
