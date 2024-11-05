from typing import Type, Sequence, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t

from .ModelAttackerLSTMLinear import ModelAttackerLSTMLinear
from utils_torch.modules import MultiLayerConv1d, hx_type

_activations = [
    "Identity",
    "Threshold",
    "ReLU",
    "RReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Hardsigmoid",
    "Tanh",
    "SiLU",
    "Mish",
    "Hardswish",
    "ELU",
    "CELU",
    "SELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "MultiheadAttention",
    "PReLU",
    "Softsign",
    "Tanhshrink",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
]

class ModelAttackerConvLSTMLinear(ModelAttackerLSTMLinear):
    """
    A neural network model that combines a 1D convolutional layer with an LSTM and a linear layer.
    Inherits from ModelAttackerLSTMLinear.

    Attributes:
        conv (nn.Conv1d): The 1D convolutional layer.
    """

    _conv_lstm_activation: Type[nn.Module]
    _conv_lstm_activation_kwargs: dict[str, any]

    _conv_lstm_linear_dropout_p: float
    _conv_lstm_linear_dropout_first: bool

    _conv_lstm_layer_norm: bool
    _conv_lstm_layer_norm_element_wise_affine: bool

    conv_modules: MultiLayerConv1d
    conv_lstm_activation_module: nn.Module
    conv_lstm_dropout_module: nn.Module
    conv_lstm_layer_norm_module: nn.Module

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
                 conv_batch_norm_affine: bool | Sequence[bool] = True,
                 conv_batch_norm_track_running_stats: bool | Sequence[bool] = True,
                 conv_lstm_activation: Type[nn.Module] = nn.ReLU,
                 conv_lstm_activation_kwargs: dict[str, any] = None,
                 conv_lstm_dropout_p: float = 0.5,
                 conv_lstm_dropout_first: bool = True,
                 conv_lstm_layer_norm: bool = True,
                 conv_lstm_layer_norm_element_wise_affine: bool = True,
                 lstm_proj_size: int | Sequence[int] = 0,
                 lstm_bidirectional: bool | Sequence[bool] = False,
                 lstm_dropout_p: float | Sequence[float] = 0.5,
                 lstm_dropout_first: bool | Sequence[bool] = True,
                 lstm_layer_norm: bool | Sequence[bool] = True,
                 lstm_layer_norm_element_wise_affine: bool | Sequence[bool] = True,
                 lstm_linear_dropout_p: float = 0.25,
                 lstm_linear_dropout_first: bool = True,
                 lstm_linear_batch_norm: bool = True,
                 lstm_linear_batch_norm_momentum: float = 0.1,
                 lstm_linear_batch_norm_affine: bool = True,
                 lstm_linear_batch_norm_track_running_stats: bool = True,
                 linear_activation: Type[nn.Module] = nn.ReLU,
                 linear_activation_kwargs: dict[str, any] = None,
                 linear_dropout_p: float = 0.5,
                 linear_dropout_first: bool = True,
                 linear_batch_norm: bool = True,
                 linear_batch_norm_momentum: float = 0.1,
                 linear_batch_norm_affine: bool = True,
                 linear_batch_norm_track_running_stats: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes the ModelAttackerConvLSTMLinear.

        Args:
            conv_in_channels (int): Number of input channels for the convolutional layer.
            conv_out_channels (int): Number of output channels for the convolutional layer.
            conv_kernel_size (int): Kernel size for the convolutional layer.
            conv_stride (int): Stride for the convolutional layer.
            lstm_hidden_size (int): Number of features in the hidden state of the LSTM.
            lstm_num_layers (int, optional): Number of recurrent layers in the LSTM. Defaults to 1.
            lstm_bidirectional (bool, optional): If True, the LSTM is bidirectional. Defaults to False.
            lstm_dropout (float, optional): Dropout probability for the LSTM. Defaults to 0.5.
            linear_out_features (int, optional): Number of output features for the linear layer. Defaults to 1.
        """
        super(ModelAttackerConvLSTMLinear, self).__init__(lstm_num_layers=lstm_num_layers,
                                                          lstm_input_size=lstm_input_size,
                                                          lstm_hidden_size=lstm_hidden_size,
                                                          linear_num_layers=linear_num_layers,
                                                          linear_num_features=linear_num_features,
                                                          lstm_proj_size=lstm_proj_size,
                                                          lstm_bidirectional=lstm_bidirectional,
                                                          lstm_dropout_p=lstm_dropout_p,
                                                          lstm_dropout_first=lstm_dropout_first,
                                                          lstm_layer_norm=lstm_layer_norm,
                                                          lstm_layer_norm_element_wise_affine=lstm_layer_norm_element_wise_affine,
                                                          lstm_linear_dropout_p=lstm_linear_dropout_p,
                                                          lstm_linear_dropout_first=lstm_linear_dropout_first,
                                                          lstm_linear_batch_norm=lstm_linear_batch_norm,
                                                          lstm_linear_batch_norm_momentum=lstm_linear_batch_norm_momentum,
                                                          lstm_linear_batch_norm_affine=lstm_linear_batch_norm_affine,
                                                          lstm_linear_batch_norm_track_running_stats=lstm_linear_batch_norm_track_running_stats,
                                                          linear_activation=linear_activation,
                                                          linear_activation_kwargs=linear_activation_kwargs,
                                                          linear_dropout_p=linear_dropout_p,
                                                          linear_dropout_first=linear_dropout_first,
                                                          linear_batch_norm=linear_batch_norm,
                                                          linear_batch_norm_momentum=linear_batch_norm_momentum,
                                                          linear_batch_norm_affine=linear_batch_norm_affine,
                                                          linear_batch_norm_track_running_stats=linear_batch_norm_track_running_stats,
                                                          device=device,
                                                          dtype=dtype)

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
                                             batch_norm_affine=conv_batch_norm_affine,
                                             batch_norm_track_running_stats=conv_batch_norm_track_running_stats,
                                             device=device,
                                             dtype=dtype)

        assert self.conv_modules.channel_size_out == self.lstm_input_size_in, "The number of output channels from the convolutional layer must match the number of input features for the LSTM."

        self._conv_lstm_activation = conv_lstm_activation
        self._conv_lstm_activation_kwargs = conv_lstm_activation_kwargs

        self._conv_lstm_linear_dropout_p = conv_lstm_dropout_p
        self._conv_lstm_linear_dropout_first = conv_lstm_dropout_first

        self._conv_lstm_layer_norm = conv_lstm_layer_norm
        self._conv_lstm_layer_norm_element_wise_affine = conv_lstm_layer_norm_element_wise_affine

        if conv_lstm_activation is not None:
            self.conv_lstm_activation_module = conv_lstm_activation(**conv_lstm_activation_kwargs,
                                                             device=device,
                                                             dtype=dtype)
        else:
            self.conv_lstm_activation_module = nn.Identity()

        if conv_lstm_dropout_p > 0:
            self.conv_lstm_dropout_module = nn.Dropout(p=conv_lstm_dropout_p)
        else:
            self.conv_lstm_dropout_module = nn.Identity()

        if conv_lstm_layer_norm:
            self.conv_lstm_layer_norm_module = nn.LayerNorm(normalized_shape=self.conv_modules.conv_channel_size_out,
                                                     elementwise_affine=conv_lstm_layer_norm_element_wise_affine,
                                                     device=device,
                                                     dtype=dtype)
        else:
            self.conv_lstm_layer_norm_module = nn.Identity()

    def forward(self,
                x: Tensor,
                hx: hx_type = None) -> tuple[Tensor, hx_type]:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, conv_in_channels).
            hx (hx_type, optional): Initial hidden state. Default is None.

        Returns:
            tuple[tuple[Tensor, Tensor], Tensor]: Output from the LSTM and the final hidden and cell states.
        """
        x = x.permute(0, 2, 1) # batch, conv_in_channels, seq_len
        x = self.conv_modules(x)
        x = self.conv_lstm_activation_module(x)
        x = self.conv_lstm_dropout_module(x)
        x = x.permute(0, 2, 1) # batch, seq_len, conv_out_channels
        x = self.conv_lstm_layer_norm_module(x)
        return super().forward(x, hx)

    @property
    def conv_in_channels(self) -> int:
        """
        Returns the number of input channels for the convolutional layer.

        Returns:
            int: Number of input channels.
        """
        return self.conv_modules.in_channels

    @property
    def conv_out_channels(self) -> int:
        """
        Returns the number of output channels for the convolutional layer.

        Returns:
            int: Number of output channels.
        """
        return self.conv_modules.out_channels

    @property
    def conv_kernel_size(self) -> int:
        """
        Returns the kernel size for the convolutional layer.

        Returns:
            int: Kernel size.
        """
        return self.conv_modules.kernel_size[0]

    @property
    def conv_stride(self) -> int:
        """
        Returns the stride for the convolutional layer.

        Returns:
            int: Stride.
        """
        return self.conv_modules.stride[0]
