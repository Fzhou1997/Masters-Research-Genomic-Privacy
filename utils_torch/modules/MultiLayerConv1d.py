from typing import Sequence, Literal, Type

import torch
from torch import nn, Tensor
from torch.nn.common_types import _size_1_t


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

class MultiLayerConv1d(nn.Module):
    """
    A multi-layer 1D convolutional module with optional batch normalization and dropout.

    Attributes:
        _conv_num_layers (int): Number of convolutional layers.
        _inner_num_layers (int): Number of inner layers.

        _conv_channel_size (Sequence[int]): Number of channels in each convolutional layer.
        _conv_kernel_size (Sequence[int]): Kernel size for each convolutional layer.
        _conv_stride (Sequence[int]): Stride for each convolutional layer.
        _conv_padding (Sequence[Literal['same', 'valid'] | int | _size_1_t]): Padding for each convolutional layer.
        _conv_dilation (Sequence[int | _size_1_t]): Dilation for each convolutional layer.
        _conv_groups (Sequence[int]): Number of groups for each convolutional layer.
        _conv_bias (Sequence[bool]): Whether to use bias in each convolutional layer.
        _conv_padding_mode (Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']]): Padding mode for each convolutional layer.

        _activation (Sequence[Type[nn.Module]]): Activation function for each layer.
        _activation_kwargs (Sequence[dict[str, any]]): Keyword arguments for the activation function.

        _dropout_p (Sequence[float]): Dropout probability for each dropout layer.
        _dropout_inplace (Sequence[bool]): Whether to perform dropout in-place.
        _dropout_first (Sequence[bool]): Whether to perform dropout before the batch normalization layer.

        _batch_norm (Sequence[bool]): Whether to apply batch normalization.
        _batch_norm_eps (Sequence[float]): Epsilon value for batch normalization.
        _batch_norm_momentum (Sequence[float]): Momentum value for batch normalization.
        _batch_norm_affine (Sequence[bool]): Whether to learn affine parameters in batch normalization.
        _batch_norm_track_running_stats (Sequence[bool]): Whether to track running statistics in batch normalization.

        _multi_layer_modules (nn.Sequential): Sequential container of layers.
    """

    _conv_num_layers: int
    _inner_num_layers: int

    _conv_channel_size: tuple[int, ...]
    _conv_kernel_size: tuple[int, ...]
    _conv_stride: tuple[int, ...]
    _conv_padding: tuple[Literal['same', 'valid'] | int | _size_1_t, ...]
    _conv_dilation: tuple[int | _size_1_t, ...]
    _conv_groups: tuple[int, ...]
    _conv_bias: tuple[bool, ...]
    _conv_padding_mode: tuple[Literal['zeros', 'reflect', 'replicate', 'circular'], ...]

    _activation: tuple[Type[nn.Module], ...]
    _activation_kwargs: tuple[dict[str, any], ...]

    _dropout_p: tuple[float, ...]
    _dropout_inplace: tuple[bool, ...]
    _dropout_first: tuple[bool, ...]

    _batch_norm: tuple[bool, ...]
    _batch_norm_eps: tuple[float, ...]
    _batch_norm_momentum: tuple[float, ...]
    _batch_norm_affine: tuple[bool, ...]
    _batch_norm_track_running_stats: tuple[bool, ...]

    _multi_layer_modules: nn.Sequential

    def __init__(self,
                 num_layers: int,
                 channel_size: int | Sequence[int],
                 kernel_size: int | Sequence[int],
                 stride: int | Sequence[int] = 1,
                 padding: Literal['same', 'valid'] | int | _size_1_t | Sequence[Literal['same', 'valid'] | int | _size_1_t] = 0,
                 dilation: int | _size_1_t | Sequence[int | _size_1_t] = 1,
                 groups: int | Sequence[int] = 1,
                 bias: bool | Sequence[bool] = True,
                 padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] | Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']] = 'zeros',
                 activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 activation_kwargs: dict[str, any] | Sequence[dict[str, any]] = None,
                 dropout_p: float | Sequence[float] = 0.5,
                 dropout_inplace: bool | Sequence[bool] = True,
                 dropout_first: bool | Sequence[bool] = True,
                 batch_norm: bool | Sequence[bool] = True,
                 batch_norm_eps: float | Sequence[float] = 1e-5,
                 batch_norm_momentum: float | Sequence[float] = 0.1,
                 batch_norm_affine: bool | Sequence[bool] = True,
                 batch_norm_track_running_stats: bool | Sequence[bool] = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes the ModuleMultiLayerConv1d.

        Args:
            num_layers (int): Number of layers.
            channel_size (int | Sequence[int]): Number of channels in each layer.
            kernel_size (int | Sequence[int]): Kernel size for each layer.
            stride (int | Sequence[int], optional): Stride for each layer, default is 1.
            padding (Literal['same', 'valid'] | int | _size_1_t | Sequence[Literal['same', 'valid'] | int | _size_1_t], optional): Padding for each layer, default is 0.
            dilation (int | _size_1_t | Sequence[int | _size_1_t], optional): Dilation for each layer, default is 1.
            groups (int | Sequence[int], optional): Number of groups for each layer, default is 1.
            bias (bool | Sequence[bool], optional): Whether to use bias in each layer, default is True.
            padding_mode (Literal['zeros', 'reflect', 'replicate', 'circular'] | Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']], optional): Padding mode for each layer, default is 'zeros'.
            activation (Type[nn.Module] | Sequence[Type[nn.Module]], optional): Activation function for each layer, default is nn.ReLU.
            activation_kwargs (dict[str, Any] | Sequence[dict[str, Any]], optional): Keyword arguments for the activation function, default is None.
            dropout_p (float | Sequence[float], optional): Dropout probability for each dropout layer, default is 0.5.
            dropout_inplace (bool | Sequence[bool], optional): Whether to perform dropout in-place, default is False.
            dropout_first (bool | Sequence[bool], optional): Whether to perform dropout before the batch normalization layer, default is True.
            batch_norm (bool | Sequence[bool], optional): Whether to apply batch normalization, default is True.
            batch_norm_eps (float | Sequence[float], optional): Epsilon value for batch normalization, default is 1e-5.
            batch_norm_momentum (float | Sequence[float], optional): Momentum value for batch normalization, default is 0.1.
            batch_norm_affine (bool | Sequence[bool], optional): Whether to learn affine parameters in batch normalization, default is True.
            batch_norm_track_running_stats (bool | Sequence[bool], optional): Whether to track running statistics in batch normalization, default is True.
            device (torch.device, optional): Device for the tensors, default is None.
            dtype (torch.dtype, optional): Data type for the tensors, default is None.

        Raises:
            AssertionError: If num_layers is less than or equal to 0.
            AssertionError: If channel_size does not have length num_layers + 1.
            AssertionError: If kernel_size, stride, padding, dilation, groups, bias, and padding_mode do not have length num_layers.
            AssertionError: If activation, activation_kwargs do not have length num_layers - 1.
            AssertionError: If activation is not one of the supported activation functions.
            AssertionError: If dropout_p, dropout_inplace, dropout_first do not have length num_layers - 1.
            AssertionError: If batch_norm, batch_norm_eps, batch_norm_momentum, batch_norm_affine, and batch_norm_track_running_stats do not have length num_layers - 1.
        """

        super(MultiLayerConv1d, self).__init__()

        assert num_layers > 0, 'num_layers must be greater than 0'

        self._conv_num_layers = num_layers
        self._inner_num_layers = num_layers - 1

        if isinstance(channel_size, int):
            channel_size = [channel_size] * (self._conv_num_layers + 1)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * self._conv_num_layers
        if isinstance(stride, int):
            stride = [stride] * self._conv_num_layers
        if isinstance(padding, (int, str)):
            padding = [padding] * self._conv_num_layers
        if isinstance(dilation, int):
            dilation = [dilation] * self._conv_num_layers
        if isinstance(groups, int):
            groups = [groups] * self._conv_num_layers
        if isinstance(bias, bool):
            bias = [bias] * self._conv_num_layers
        if isinstance(padding_mode, str):
            padding_mode = [padding_mode] * self._conv_num_layers

        if isinstance(activation, type):
            activation = [activation] * self._inner_num_layers
        if isinstance(activation_kwargs, dict):
            activation_kwargs = [activation_kwargs] * self._inner_num_layers
        elif activation_kwargs is None:
            activation_kwargs = [{}] * self._inner_num_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * self._inner_num_layers
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * self._inner_num_layers
        if isinstance(dropout_first, bool):
            dropout_first = [dropout_first] * self._inner_num_layers

        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * self._inner_num_layers
        if isinstance(batch_norm_eps, float):
            batch_norm_eps = [batch_norm_eps] * self._inner_num_layers
        if isinstance(batch_norm_momentum, float):
            batch_norm_momentum = [batch_norm_momentum] * self._inner_num_layers
        if isinstance(batch_norm_affine, bool):
            batch_norm_affine = [batch_norm_affine] * self._inner_num_layers
        if isinstance(batch_norm_track_running_stats, bool):
            batch_norm_track_running_stats = [batch_norm_track_running_stats] * self._inner_num_layers

        assert len(channel_size) == num_layers + 1, 'channel_size must have length num_layers + 1'
        assert len(kernel_size) == num_layers, 'kernel_size must have length num_layers'
        assert len(stride) == num_layers, 'stride must have length num_layers'
        assert len(padding) == num_layers, 'padding must have length num_layers'
        assert len(dilation) == num_layers, 'dilation must have length num_layers'
        assert len(groups) == num_layers, 'groups must have length num_layers'
        assert len(bias) == num_layers, 'bias must have length num_layers'
        assert len(padding_mode) == num_layers, 'padding_mode must have length num_layers'

        assert len(activation) == num_layers - 1, 'activation must have length num_layers - 1'
        assert len(activation_kwargs) == num_layers - 1, 'activation_kwargs must have length num_layers - 1'
        assert all([cls is None or cls.__name__ in _activations for cls in activation]), 'activation must be one of the following types: ' + ', '.join(_activations)

        assert len(dropout_p) == num_layers - 1, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == num_layers - 1, 'dropout_inplace must have length num_layers - 1'
        assert len(dropout_first) == num_layers - 1, 'dropout_first must have length num_layers - 1'

        assert len(batch_norm) == num_layers - 1, 'batch_norm must have length num_layers - 1'
        assert len(batch_norm_eps) == num_layers - 1, 'batch_norm_eps must have length num_layers - 1'
        assert len(batch_norm_momentum) == num_layers - 1, 'batch_norm_momentum must have length num_layers - 1'
        assert len(batch_norm_affine) == num_layers - 1, 'batch_norm_affine must have length num_layers - 1'
        assert len(batch_norm_track_running_stats) == num_layers - 1, 'batch_norm_track_running_stats must have length num_layers - 1'

        self._conv_channel_size = tuple(channel_size)
        self._conv_kernel_size = tuple(kernel_size)
        self._conv_stride = tuple(stride)
        self._conv_padding = tuple(padding)
        self._conv_dilation = tuple(dilation)
        self._conv_groups = tuple(groups)
        self._conv_bias = tuple(bias)
        self._conv_padding_mode = tuple(padding_mode)

        self._activation = tuple(activation)
        self._activation_kwargs = tuple(activation_kwargs)

        self._dropout_p = tuple(dropout_p)
        self._dropout_inplace = tuple(dropout_inplace)
        self._dropout_first = tuple(dropout_first)

        self._batch_norm = tuple(batch_norm)
        self._batch_norm_eps = tuple(batch_norm_eps)
        self._batch_norm_momentum = tuple(batch_norm_momentum)
        self._batch_norm_affine = tuple(batch_norm_affine)
        self._batch_norm_track_running_stats = tuple(batch_norm_track_running_stats)

        self._multi_layer_modules = nn.Sequential()

        for i in range(num_layers):
            self._multi_layer_modules.append(nn.Conv1d(in_channels=channel_size[i],
                                                       out_channels=channel_size[i + 1],
                                                       kernel_size=kernel_size[i],
                                                       stride=stride[i],
                                                       padding=padding[i],
                                                       dilation=dilation[i],
                                                       groups=groups[i],
                                                       bias=bias[i],
                                                       device=device,
                                                       dtype=dtype))
            if i >= self._inner_num_layers:
                continue
            self._multi_layer_modules.append(activation[i](**activation_kwargs[i]))
            if dropout_first[i]:
                if dropout_p[i] > 0:
                    self._multi_layer_modules.append(nn.Dropout(p=dropout_p[i],
                                                                inplace=dropout_inplace[i]))
                if batch_norm[i]:
                    self._multi_layer_modules.append(nn.BatchNorm1d(num_features=channel_size[i + 1],
                                                                    eps=batch_norm_eps[i],
                                                                    momentum=batch_norm_momentum[i],
                                                                    affine=batch_norm_affine[i],
                                                                    track_running_stats=batch_norm_track_running_stats[i],
                                                                    device=device,
                                                                    dtype=dtype))
            else:
                if batch_norm[i]:
                    self._multi_layer_modules.append(nn.BatchNorm1d(num_features=channel_size[i + 1],
                                                                    eps=batch_norm_eps[i],
                                                                    momentum=batch_norm_momentum[i],
                                                                    affine=batch_norm_affine[i],
                                                                    track_running_stats=batch_norm_track_running_stats[i],
                                                                    device=device,
                                                                    dtype=dtype))
                if dropout_p[i] > 0:
                    self._multi_layer_modules.append(nn.Dropout(p=dropout_p[i],
                                                                inplace=dropout_inplace[i]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the layers.
        """
        return self._multi_layer_modules(x)

    @property
    def conv_num_layers(self) -> int:
        """
        Returns the number of convolutional layers.

        Returns:
            int: Number of convolutional layers.
        """
        return self._conv_num_layers

    @property
    def conv_channel_size_in(self) -> int:
        """
        Returns the number of input channels for the first convolutional layer.

        Returns:
            int: Number of input channels.
        """
        return self._conv_channel_size[0]

    @property
    def conv_channel_size_out(self) -> int:
        """
        Returns the number of output channels for the last convolutional layer.

        Returns:
            int: Number of output channels.
        """
        return self._conv_channel_size[-1]

    @property
    def conv_channel_size(self) -> tuple[int, ...]:
        """
        Returns the number of channels for each convolutional layer.

        Returns:
            tuple[int, ...]: Number of channels for each convolutional layer.
        """
        return self._conv_channel_size

    @property
    def conv_kernel_size(self) -> tuple[int, ...]:
        """
        Returns the kernel size for each convolutional layer.

        Returns:
            tuple[int, ...]: Kernel size for each convolutional layer.
        """
        return self._conv_kernel_size

    @property
    def conv_stride(self) -> tuple[int, ...]:
        """
        Returns the stride for each convolutional layer.

        Returns:
            tuple[int, ...]: Stride for each convolutional layer.
        """
        return self._conv_stride

    @property
    def conv_padding(self) -> tuple[Literal['same', 'valid'] | int | _size_1_t, ...]:
        """
        Returns the padding for each convolutional layer.

        Returns:
            tuple[Literal['same', 'valid'] | int | _size_1_t, ...]: Padding for each convolutional layer.
        """
        return self._conv_padding

    @property
    def conv_dilation(self) -> tuple[int | _size_1_t, ...]:
        """
        Returns the dilation for each convolutional layer.

        Returns:
            tuple[int | _size_1_t, ...]: Dilation for each convolutional layer.
        """
        return self._conv_dilation

    @property
    def conv_groups(self) -> tuple[int, ...]:
        """
        Returns the number of groups for each convolutional layer.

        Returns:
            tuple[int, ...]: Number of groups for each convolutional layer.
        """
        return self._conv_groups

    @property
    def conv_bias(self) -> tuple[bool, ...]:
        """
        Returns whether to use bias for each convolutional layer.

        Returns:
            tuple[bool, ...]: Whether to use bias for each convolutional layer.
        """
        return self._conv_bias

    @property
    def conv_padding_mode(self) -> tuple[Literal['zeros', 'reflect', 'replicate', 'circular'], ...]:
        """
        Returns the padding mode for each convolutional layer.

        Returns:
            tuple[Literal['zeros', 'reflect', 'replicate', 'circular'], ...]: Padding mode for each convolutional layer.
        """
        return self._conv_padding_mode

    @property
    def conv_activation(self) -> tuple[Type[nn.Module], ...]:
        """
        Returns the activation function for each layer.

        Returns:
            tuple[Type[nn.Module], ...]: Activation function for each layer.
        """
        return self._activation

    @property
    def conv_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        """
        Returns the keyword arguments for the activation function.

        Returns:
            tuple[dict[str, any], ...]: Keyword arguments for the activation function.
        """
        return self._activation_kwargs

    @property
    def conv_dropout_p(self) -> tuple[float, ...]:
        """
        Returns the dropout probability for each dropout layer.

        Returns:
            tuple[float, ...]: Dropout probability for each dropout layer.
        """
        return self._dropout_p

    @property
    def conv_dropout_inplace(self) -> tuple[bool, ...]:
        """
        Returns whether to perform dropout in-place.

        Returns:
            tuple[bool, ...]: Whether to perform dropout in-place.
        """
        return self._dropout_inplace

    @property
    def conv_dropout_first(self) -> tuple[bool, ...]:
        """
        Returns whether to perform dropout before the batch normalization layer.

        Returns:
            tuple[bool, ...]: Whether to perform dropout before the batch normalization layer.
        """
        return self._dropout_first

    @property
    def conv_batch_norm(self) -> tuple[bool, ...]:
        """
        Returns whether to apply batch normalization.

        Returns:
            tuple[bool, ...]: Whether to apply batch normalization.
        """
        return self._batch_norm

    @property
    def conv_batch_norm_eps(self) -> tuple[float, ...]:
        """
        Returns the epsilon value for batch normalization.

        Returns:
            tuple[float, ...]: Epsilon value for batch normalization.
        """
        return self._batch_norm_eps

    @property
    def conv_batch_norm_momentum(self) -> tuple[float, ...]:
        """
        Returns the momentum value for batch normalization.

        Returns:
            tuple[float, ...]: Momentum value for batch normalization.
        """
        return self._batch_norm_momentum

    @property
    def conv_batch_norm_affine(self) -> tuple[bool, ...]:
        """
        Returns whether to learn affine parameters in batch normalization.

        Returns:
            tuple[bool, ...]: Whether to learn affine parameters in batch normalization.
        """
        return self._batch_norm_affine

    @property
    def conv_batch_norm_track_running_stats(self) -> tuple[bool, ...]:
        """
        Returns whether to track running statistics in batch normalization.

        Returns:
            tuple[bool, ...]: Whether to track running statistics in batch normalization.
        """
        return self._batch_norm_track_running_stats

    def get_seq_size_out(self, seq_size_in: int) -> int:
        seq_size_out = seq_size_in
        for i in range(self.conv_num_layers):
            seq_size_out = (seq_size_out + 2 * self.conv_padding[i] - self.conv_dilation[i] * (self.conv_kernel_size[i] - 1) - 1) // self.conv_stride[i] + 1
        return seq_size_out
