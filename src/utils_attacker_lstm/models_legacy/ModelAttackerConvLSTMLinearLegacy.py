from typing import Type, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t

from src.utils_torch.modules_legacy.MultiLayerHiddenCellLSTM import hx_type, y_type
from src.utils_torch.modules_legacy import MultiLayerConv1d
from .ModelAttackerLSTMLinearLegacy import ModelAttackerLSTMLinearLegacy


class ModelAttackerConvLSTMLinearLegacy(ModelAttackerLSTMLinearLegacy):
    """
    A neural network model that combines a 1D convolutional layer with an LSTM and a linear layer.
    Inherits from ModelAttackerLSTMLinear.

    Attributes:
        _conv_lstm_activation (Type[nn.Module]): Activation function between the convolutional and LSTM layers.
        _conv_lstm_activation_kwargs (dict[str, any]): Activation function keyword arguments between the convolutional and LSTM layers.
        _conv_lstm_linear_dropout_p (float): Dropout probability between the convolutional and linear layers.
        _conv_lstm_linear_dropout_first (bool): If True, dropout is applied before the linear layer.
        _conv_lstm_layer_norm (bool): If True, layer normalization is used between the convolutional and LSTM layers.
        conv_modules (MultiLayerConv1d | nn.Identity): The convolutional layers used in the model.
        conv_lstm_activation_module (nn.Module): The activation module between the convolutional and LSTM layers.
        conv_lstm_dropout_module (nn.Module | nn.Identity): The dropout module between the convolutional and LSTM layers.
        conv_lstm_layer_norm_module (nn.Module | nn.Identity): The layer normalization module between the convolutional and LSTM layers.
    """

    _conv_lstm_activation: Type[nn.Module]
    _conv_lstm_activation_kwargs: dict[str, any]

    _conv_lstm_linear_dropout_p: float
    _conv_lstm_linear_dropout_first: bool

    _conv_lstm_layer_norm: bool

    conv_modules: MultiLayerConv1d | nn.Identity
    conv_lstm_activation_module: nn.Module
    conv_lstm_dropout_module: nn.Dropout | nn.Identity
    conv_lstm_layer_norm_module: nn.LayerNorm | nn.Identity

    def __init__(self,
                 conv_num_layers: int,
                 conv_channel_size: int | Sequence[int],
                 conv_kernel_size: int | Sequence[int],
                 lstm_num_layers: int,
                 lstm_input_size: int,
                 lstm_hidden_size: int | Sequence[int],
                 linear_num_layers: int,
                 linear_num_features: int | Sequence[int],
                 conv_stride: int | Sequence[int] = 1,
                 conv_dilation: int | _size_1_t | Sequence[int | _size_1_t] = 1,
                 conv_groups: int | Sequence[int] = 1,
                 conv_activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 conv_activation_kwargs: dict[str, any] | Sequence[dict[str, any]] = None,
                 conv_dropout_p: float | Sequence[float] = 0.5,
                 conv_dropout_first: bool | Sequence[bool] = True,
                 conv_batch_norm: bool | Sequence[bool] = True,
                 conv_batch_norm_momentum: float | Sequence[float] = 0.1,
                 conv_lstm_activation: Type[nn.Module] = nn.ReLU,
                 conv_lstm_activation_kwargs: dict[str, any] = None,
                 conv_lstm_dropout_p: float = 0.5,
                 conv_lstm_dropout_first: bool = True,
                 conv_lstm_layer_norm: bool = True,
                 lstm_proj_size: int | Sequence[int] = 0,
                 lstm_bidirectional: bool | Sequence[bool] = False,
                 lstm_dropout_p: float | Sequence[float] = 0.5,
                 lstm_dropout_first: bool | Sequence[bool] = True,
                 lstm_layer_norm: bool | Sequence[bool] = True,
                 lstm_linear_dropout_p: float = 0.25,
                 lstm_linear_dropout_first: bool = True,
                 lstm_linear_batch_norm: bool = True,
                 lstm_linear_batch_norm_momentum: float = 0.1,
                 linear_activation: Type[nn.Module] = nn.ReLU,
                 linear_activation_kwargs: dict[str, any] = None,
                 linear_dropout_p: float = 0.5,
                 linear_dropout_first: bool = True,
                 linear_batch_norm: bool = True,
                 linear_batch_norm_momentum: float = 0.1,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes the ModelAttackerConvLSTMLinear.

        Args:
            conv_num_layers (int): Number of convolutional layers.
            conv_channel_size (int | Sequence[int]): Number of input channels for the first convolutional layer.
            conv_kernel_size (int | Sequence[int]): Kernel size for each convolutional layer.
            lstm_num_layers (int): Number of recurrent layers.
            lstm_input_size (int): The number of expected features in the input.
            lstm_hidden_size (int | Sequence[int]): The number of features in the hidden state.
            linear_num_layers (int): Number of linear layers.
            linear_num_features (int | Sequence[int]): Number of features in the linear layer.
            conv_stride (int | Sequence[int], optional): Stride for each convolutional layer. Default is 1.
            conv_dilation (int | _size_1_t | Sequence[int | _size_1_t], optional): Dilation for each convolutional layer. Default is 1.
            conv_groups (int | Sequence[int], optional): Number of groups for each convolutional layer. Default is 1.
            conv_activation (Type[nn.Module] | Sequence[Type[nn.Module]], optional): Activation function for each convolutional layer. Default is nn.ReLU.
            conv_activation_kwargs (dict[str, any] | Sequence[dict[str, any]], optional): Activation function keyword arguments for each convolutional layer. Default is None.
            conv_dropout_p (float | Sequence[float], optional): Dropout probability for each convolutional layer. Default is 0.5.
            conv_dropout_first (bool | Sequence[bool], optional): Dropout first for each convolutional layer. Default is True.
            conv_batch_norm (bool | Sequence[bool], optional): If True, batch normalization is used for each convolutional layer. Default is True.
            conv_batch_norm_momentum (float | Sequence[float], optional): Momentum for each batch normalization layer. Default is 0.1.
            conv_lstm_activation (Type[nn.Module], optional): Activation function for the convolutional layer. Default is nn.ReLU.
            conv_lstm_activation_kwargs (dict[str, any], optional): Activation function keyword arguments for the convolutional layer. Default is None.
            conv_lstm_dropout_p (float, optional): Dropout probability for the convolutional layer. Default is 0.5.
            conv_lstm_dropout_first (bool, optional): Dropout first for the convolutional layer. Default is True.
            conv_lstm_layer_norm (bool, optional): If True, layer normalization is used for the convolutional layer. Default is True.
            lstm_proj_size (int | Sequence[int], optional): The number of features in the projected state. Default is 0.
            lstm_bidirectional (bool | Sequence[bool], optional): If True, becomes a bidirectional LSTM. Default is False.
            lstm_dropout_p (float | Sequence[float], optional): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer. Default is 0.5.
            lstm_dropout_first (bool | Sequence[bool], optional): If True, dropout is applied before the LSTM layer. Default is True.
            lstm_layer_norm (bool | Sequence[bool], optional): If True, layer normalization is used for each LSTM layer. Default is True.
            lstm_linear_dropout_p (float, optional): Dropout probability for the linear layer. Default is 0.25.
            lstm_linear_dropout_first (bool, optional): If True, dropout is applied before the linear layer. Default is True.
            lstm_linear_batch_norm (bool, optional): If True, batch normalization is used for the linear layer. Default is True.
            lstm_linear_batch_norm_momentum (float, optional): Momentum for the batch normalization layer. Default is 0.1.
            linear_activation (Type[nn.Module], optional): Activation function for the linear layer. Default is nn.ReLU.
            linear_activation_kwargs (dict[str, any], optional): Activation function keyword arguments for the linear layer. Default is None.
            linear_dropout_p (float, optional): Dropout probability for the linear layer. Default is 0.5.
            linear_dropout_first (bool, optional): If True, dropout is applied before the linear layer. Default is True.
            linear_batch_norm (bool, optional): If True, batch normalization is used for the linear layer. Default is True.
            linear_batch_norm_momentum (float, optional): Momentum for the batch normalization layer. Default is 0.1.
            device (torch.device, optional): The desired device of the model. Default is None.
            dtype (torch.dtype, optional): The desired data type of the model. Default is None.
        """
        super(ModelAttackerConvLSTMLinearLegacy, self).__init__(lstm_num_layers=lstm_num_layers,
                                                                lstm_input_size=lstm_input_size,
                                                                lstm_hidden_size=lstm_hidden_size,
                                                                linear_num_layers=linear_num_layers,
                                                                linear_num_features=linear_num_features,
                                                                lstm_proj_size=lstm_proj_size,
                                                                lstm_bidirectional=lstm_bidirectional,
                                                                lstm_dropout_p=lstm_dropout_p,
                                                                lstm_dropout_first=lstm_dropout_first,
                                                                lstm_layer_norm=lstm_layer_norm,
                                                                lstm_linear_dropout_p=lstm_linear_dropout_p,
                                                                lstm_linear_dropout_first=lstm_linear_dropout_first,
                                                                lstm_linear_batch_norm=lstm_linear_batch_norm,
                                                                lstm_linear_batch_norm_momentum=lstm_linear_batch_norm_momentum,
                                                                linear_activation=linear_activation,
                                                                linear_activation_kwargs=linear_activation_kwargs,
                                                                linear_dropout_p=linear_dropout_p,
                                                                linear_dropout_first=linear_dropout_first,
                                                                linear_batch_norm=linear_batch_norm,
                                                                linear_batch_norm_momentum=linear_batch_norm_momentum,
                                                                device=device,
                                                                dtype=dtype)

        assert conv_num_layers >= 0, "The number of convolutional layers must be greater than or equal to 0."

        if conv_num_layers == 0:
            self.conv_modules = nn.Identity()
            self.conv_lstm_activation_module = nn.Identity()
            self.conv_lstm_dropout_module = nn.Identity()
            self.conv_lstm_layer_norm_module = nn.Identity()
            return

        self.conv_modules = MultiLayerConv1d(num_layers=conv_num_layers,
                                             channel_size=conv_channel_size,
                                             kernel_size=conv_kernel_size,
                                             stride=conv_stride,
                                             dilation=conv_dilation,
                                             groups=conv_groups,
                                             activation=conv_activation,
                                             activation_kwargs=conv_activation_kwargs,
                                             dropout_p=conv_dropout_p,
                                             dropout_first=conv_dropout_first,
                                             batch_norm=conv_batch_norm,
                                             batch_norm_momentum=conv_batch_norm_momentum,
                                             device=device,
                                             dtype=dtype)

        assert self.conv_modules.conv_channel_size_out == self.lstm_input_size_in, "The number of output channels from the convolutional layer must match the number of input features for the LSTM."

        self._conv_lstm_activation = conv_lstm_activation
        self._conv_lstm_activation_kwargs = conv_lstm_activation_kwargs

        self._conv_lstm_linear_dropout_p = conv_lstm_dropout_p
        self._conv_lstm_linear_dropout_first = conv_lstm_dropout_first

        self._conv_lstm_layer_norm = conv_lstm_layer_norm

        if conv_lstm_activation is not None:
            self.conv_lstm_activation_module = conv_lstm_activation(**conv_lstm_activation_kwargs)
        else:
            self.conv_lstm_activation_module = nn.Identity()

        if conv_lstm_dropout_p > 0:
            self.conv_lstm_dropout_module = nn.Dropout(p=conv_lstm_dropout_p)
        else:
            self.conv_lstm_dropout_module = nn.Identity()

        if conv_lstm_layer_norm:
            self.conv_lstm_layer_norm_module = nn.LayerNorm(normalized_shape=self.conv_modules.conv_channel_size_out,
                                                            device=device,
                                                            dtype=dtype)
        else:
            self.conv_lstm_layer_norm_module = nn.Identity()

    def forward(self,
                x: Tensor,
                hx: hx_type = None) -> tuple[Tensor, y_type]:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, conv_in_channels).
            hx (hx_type, optional): Initial hidden state. Default is None.

        Returns:
            tuple[tuple[Tensor, Tensor], Tensor]: Output from the LSTM and the final hidden and cell states.
        """
        x = x.permute(0, 2, 1)  # batch, conv_in_channels, seq_len
        x = self.conv_modules(x)
        x = self.conv_lstm_activation_module(x)
        x = self.conv_lstm_dropout_module(x)
        x = x.permute(0, 2, 1)  # batch, seq_len, conv_out_channels
        x = self.conv_lstm_layer_norm_module(x)
        return super().forward(x, hx)

    @property
    def conv_num_layers(self) -> int:
        """
        Returns the number of convolutional layers.

        Returns:
            int: Number of convolutional layers.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return 0
        return self.conv_modules.conv_num_layers

    @property
    def conv_channel_size_in(self) -> int:
        """
        Returns the number of input channels for the first convolutional layer.

        Returns:
            int: Number of input channels.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return self.lstm_input_size_in
        return self.conv_modules.conv_channel_size_in

    @property
    def conv_channel_size_out(self) -> int:
        """
        Returns the number of output channels for the convolutional layer.

        Returns:
            int: Number of output channels.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return self.lstm_input_size_in
        return self.conv_modules.conv_channel_size_out

    @property
    def conv_channel_size(self) -> tuple[int, ...]:
        """
        Returns the number of output channels for each convolutional layer.

        Returns:
            tuple[int, ...]: Number of output channels.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_channel_size

    @property
    def conv_kernel_size(self) -> tuple[int, ...]:
        """
        Returns the kernel size for each convolutional layer.

        Returns:
            tuple[int, ...]: Kernel size.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_kernel_size

    @property
    def conv_stride(self) -> tuple[int, ...]:
        """
        Returns the stride for each convolutional layer.

        Returns:
            tuple[int, ...]: Stride.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_stride

    @property
    def conv_dilation(self) -> tuple[int, ...]:
        """
        Returns the dilation for each convolutional layer.

        Returns:
            tuple[int, ...]: Dilation.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_dilation

    @property
    def conv_groups(self) -> tuple[int, ...]:
        """
        Returns the number of groups for each convolutional layer.

        Returns:
            tuple[int, ...]: Number of groups.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_groups

    @property
    def conv_activation(self) -> tuple[Type[nn.Module], ...]:
        """
        Returns the activation function for each convolutional layer.

        Returns:
            tuple[Type[nn.Module], ...]: Activation function.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_activation

    @property
    def conv_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        """
        Returns the activation function keyword arguments for each convolutional layer.

        Returns:
            tuple[dict[str, any], ...]: Activation function keyword arguments.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_activation_kwargs

    @property
    def conv_dropout_p(self) -> tuple[float, ...]:
        """
        Returns the dropout probability for each convolutional layer.

        Returns:
            tuple[float, ...]: Dropout probability.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_dropout_p

    @property
    def conv_dropout_first(self) -> tuple[bool, ...]:
        """
        Returns the dropout first for each convolutional layer.

        Returns:
            tuple[bool, ...]: Dropout first.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_dropout_first

    @property
    def conv_batch_norm(self) -> tuple[bool, ...]:
        """
        Returns whether batch normalization is used for each convolutional layer.

        Returns:
            tuple[bool, ...]: True if batch normalization is used, False otherwise.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_batch_norm

    @property
    def conv_batch_norm_momentum(self) -> tuple[float, ...]:
        """
        Returns the momentum for each batch normalization layer.

        Returns:
            tuple[float, ...]: Momentum.
        """
        if isinstance(self.conv_modules, nn.Identity):
            return ()
        return self.conv_modules.conv_batch_norm_momentum

    @property
    def conv_lstm_activation(self) -> Type[nn.Module]:
        """
        Returns the activation function for the convolutional layer.

        Returns:
            Type[nn.Module]: Activation function.
        """
        return self._conv_lstm_activation

    @property
    def conv_lstm_activation_kwargs(self) -> dict[str, any]:
        """
        Returns the activation function keyword arguments for the convolutional layer.

        Returns:
            dict[str, any]: Activation function keyword arguments.
        """
        return self._conv_lstm_activation_kwargs

    @property
    def conv_lstm_dropout_p(self) -> float:
        """
        Returns the dropout probability for the convolutional layer.

        Returns:
            float: Dropout probability.
        """
        return self._conv_lstm_linear_dropout_p

    @property
    def conv_lstm_dropout_first(self) -> bool:
        """
        Returns the dropout first for the convolutional layer.

        Returns:
            bool: Dropout first.
        """
        return self._conv_lstm_linear_dropout_first

    @property
    def conv_lstm_layer_norm(self) -> bool:
        """
        Returns whether layer normalization is used for the convolutional layer.

        Returns:
            bool: True if layer normalization is used, False otherwise.
        """
        return self._conv_lstm_layer_norm
