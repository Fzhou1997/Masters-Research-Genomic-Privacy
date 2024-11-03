from typing import Sequence, Literal, Type

from torch import nn, Tensor
from torch.nn.common_types import _size_1_t


_activations = [
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
        num_conv_layers (int): Number of convolutional layers.
        num_inner_layers (int): Number of inner layers.

        conv_channel_size (Sequence[int]): Number of channels in each convolutional layer.
        conv_kernel_size (Sequence[int]): Kernel size for each convolutional layer.
        conv_stride (Sequence[int]): Stride for each convolutional layer.
        conv_padding (Sequence[Literal['same', 'valid'] | int | _size_1_t]): Padding for each convolutional layer.
        conv_dilation (Sequence[int | _size_1_t]): Dilation for each convolutional layer.
        conv_groups (Sequence[int]): Number of groups for each convolutional layer.
        conv_bias (Sequence[bool]): Whether to use bias in each convolutional layer.
        conv_padding_mode (Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']]): Padding mode for each convolutional layer.

        activation (Sequence[Type[nn.Module]]): Activation function for each layer.
        activation_kwargs (Sequence[dict[str, any]]): Keyword arguments for the activation function.

        dropout_p (Sequence[float]): Dropout probability for each dropout layer.
        dropout_inplace (Sequence[bool]): Whether to perform dropout in-place.
        dropout_first (Sequence[bool]): Whether to perform dropout before the batch normalization layer.

        batch_norm (Sequence[bool]): Whether to apply batch normalization.
        batch_norm_eps (Sequence[float]): Epsilon value for batch normalization.
        batch_norm_momentum (Sequence[float]): Momentum value for batch normalization.
        batch_norm_affine (Sequence[bool]): Whether to learn affine parameters in batch normalization.
        batch_norm_track_running_stats (Sequence[bool]): Whether to track running statistics in batch normalization.

        modules (nn.Sequential): Sequential container of layers.
    """

    num_conv_layers: int
    num_inner_layers: int

    conv_channel_size: Sequence[int]
    conv_kernel_size: Sequence[int]
    conv_stride: Sequence[int]
    conv_padding: Sequence[Literal['same', 'valid'] | int | _size_1_t]
    conv_dilation: Sequence[int | _size_1_t]
    conv_groups: Sequence[int]
    conv_bias: Sequence[bool]
    conv_padding_mode: Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']]

    activation: Sequence[Type[nn.Module]]
    activation_kwargs: Sequence[dict[str, any]]

    dropout_p: Sequence[float]
    dropout_inplace: Sequence[bool]
    dropout_first: Sequence[bool]

    batch_norm: Sequence[bool]
    batch_norm_eps: Sequence[float]
    batch_norm_momentum: Sequence[float]
    batch_norm_affine: Sequence[bool]
    batch_norm_track_running_stats: Sequence[bool]

    modules: nn.Sequential

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
                 batch_norm_track_running_stats: bool | Sequence[bool] = True) -> None:
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

        self.num_conv_layers = num_layers
        self.num_inner_layers = num_layers - 1

        if isinstance(channel_size, int):
            channel_size = [channel_size] * (self.num_conv_layers + 1)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * self.num_conv_layers
        if isinstance(stride, int):
            stride = [stride] * self.num_conv_layers
        if isinstance(padding, (int, str)):
            padding = [padding] * self.num_conv_layers
        if isinstance(dilation, int):
            dilation = [dilation] * self.num_conv_layers
        if isinstance(groups, int):
            groups = [groups] * self.num_conv_layers
        if isinstance(bias, bool):
            bias = [bias] * self.num_conv_layers
        if isinstance(padding_mode, str):
            padding_mode = [padding_mode] * self.num_conv_layers

        if isinstance(activation, type):
            activation = [activation] * self.num_inner_layers
        if isinstance(activation_kwargs, dict):
            activation_kwargs = [activation_kwargs] * self.num_inner_layers
        elif activation_kwargs is None:
            activation_kwargs = [{}] * self.num_inner_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * self.num_inner_layers
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * self.num_inner_layers
        if isinstance(dropout_first, bool):
            dropout_first = [dropout_first] * self.num_inner_layers

        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * self.num_inner_layers
        if isinstance(batch_norm_eps, float):
            batch_norm_eps = [batch_norm_eps] * self.num_inner_layers
        if isinstance(batch_norm_momentum, float):
            batch_norm_momentum = [batch_norm_momentum] * self.num_inner_layers
        if isinstance(batch_norm_affine, bool):
            batch_norm_affine = [batch_norm_affine] * self.num_inner_layers
        if isinstance(batch_norm_track_running_stats, bool):
            batch_norm_track_running_stats = [batch_norm_track_running_stats] * self.num_inner_layers

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

        self.conv_channel_size = channel_size
        self.conv_kernel_size = kernel_size
        self.conv_stride = stride
        self.conv_padding = padding
        self.conv_dilation = dilation
        self.conv_groups = groups
        self.conv_bias = bias
        self.conv_padding_mode = padding_mode

        self.activation = activation
        self.activation_kwargs = activation_kwargs

        self.dropout_p = dropout_p
        self.dropout_inplace = dropout_inplace
        self.dropout_first = dropout_first

        self.batch_norm = batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_track_running_stats = batch_norm_track_running_stats

        self.modules = nn.Sequential()

        for i in range(num_layers):
            self.modules.append(nn.Conv1d(in_channels=channel_size[i],
                                          out_channels=channel_size[i + 1],
                                          kernel_size=kernel_size[i],
                                          stride=stride[i],
                                          padding=padding[i],
                                          dilation=dilation[i],
                                          groups=groups[i],
                                          bias=bias[i]))
            if i >= self.num_inner_layers:
                continue
            self.modules.append(activation[i](**activation_kwargs[i]))
            if dropout_first[i]:
                if dropout_p[i] > 0:
                    self.modules.append(nn.Dropout(p=dropout_p[i],
                                                   inplace=dropout_inplace[i]))
                if batch_norm[i]:
                    self.modules.append(nn.BatchNorm1d(num_features=channel_size[i + 1],
                                                       eps=batch_norm_eps[i],
                                                       momentum=batch_norm_momentum[i],
                                                       affine=batch_norm_affine[i],
                                                       track_running_stats=batch_norm_track_running_stats[i]))
            else:
                if batch_norm[i]:
                    self.modules.append(nn.BatchNorm1d(num_features=channel_size[i + 1],
                                                       eps=batch_norm_eps[i],
                                                       momentum=batch_norm_momentum[i],
                                                       affine=batch_norm_affine[i],
                                                       track_running_stats=batch_norm_track_running_stats[i]))
                if dropout_p[i] > 0:
                    self.modules.append(nn.Dropout(p=dropout_p[i],
                                                   inplace=dropout_inplace[i]))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the layers.
        """
        return self.modules(x)
