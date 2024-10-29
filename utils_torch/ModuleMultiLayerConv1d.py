from typing import Sequence, Literal

from torch import nn, Tensor
from torch.nn.common_types import _size_1_t


class ModuleMultiLayerConv1d(nn.Module):
    """
    A multi-layer 1D convolutional module with optional batch normalization and dropout.

    Attributes:
        num_batch_norm_layers (int): Number of batch normalization layers.
        batch_norm (bool | Sequence[bool]): Whether to apply batch normalization.
        batch_norm_eps (float | Sequence[float]): Epsilon value for batch normalization.
        batch_norm_momentum (float | Sequence[float]): Momentum value for batch normalization.
        batch_norm_affine (bool | Sequence[bool]): Whether to learn affine parameters in batch normalization.
        batch_norm_track_running_stats (bool | Sequence[bool]): Whether to track running statistics in batch normalization.
        num_conv_layers (int): Number of convolutional layers.
        conv_channel_size (int | Sequence[int]): Number of channels in each convolutional layer.
        conv_kernel_size (int | Sequence[int]): Kernel size for each convolutional layer.
        conv_stride (int | Sequence[int]): Stride for each convolutional layer.
        conv_padding (Literal['same', 'valid'] | int | _size_1_t | Sequence[Literal['same', 'valid'] | int | _size_1_t]): Padding for each convolutional layer.
        conv_dilation (int | _size_1_t | Sequence[int | _size_1_t]): Dilation for each convolutional layer.
        conv_groups (int | Sequence[int]): Number of groups for each convolutional layer.
        conv_bias (bool | Sequence[bool]): Whether to use bias in each convolutional layer.
        conv_padding_mode (Literal['zeros', 'reflect', 'replicate', 'circular'] | Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']]): Padding mode for each convolutional layer.
        num_dropout_layers (int): Number of dropout layers.
        dropout_p (float | Sequence[float]): Dropout probability for each dropout layer.
        dropout_inplace (bool | Sequence[bool]): Whether to perform dropout in-place.
        modules (nn.Sequential): Sequential container of layers.
    """

    num_batch_norm_layers: int
    batch_norm: bool | Sequence[bool]
    batch_norm_eps: float | Sequence[float]
    batch_norm_momentum: float | Sequence[float]
    batch_norm_affine: bool | Sequence[bool]
    batch_norm_track_running_stats: bool | Sequence[bool]

    num_conv_layers: int
    conv_channel_size: int | Sequence[int]
    conv_kernel_size: int | Sequence[int]
    conv_stride: int | Sequence[int]
    conv_padding: Literal['same', 'valid'] | int | _size_1_t | Sequence[Literal['same', 'valid'] | int | _size_1_t]
    conv_dilation: int | _size_1_t | Sequence[int | _size_1_t]
    conv_groups: int | Sequence[int]
    conv_bias: bool | Sequence[bool]
    conv_padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] | Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']]

    num_dropout_layers: int
    dropout_p: float | Sequence[float]
    dropout_inplace: bool | Sequence[bool]

    modules: nn.Sequential

    def __init__(self,
                 num_layers: int,
                 channel_size: int | Sequence[int],
                 kernel_size: int | Sequence[int],
                 stride: int | Sequence[int],
                 padding: Literal['same', 'valid'] | int | _size_1_t | Sequence[Literal['same', 'valid'] | int | _size_1_t],
                 dilation: int | _size_1_t | Sequence[int | _size_1_t],
                 groups: int | Sequence[int],
                 bias: bool | Sequence[bool],
                 padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] | Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']],
                 dropout_p: float | Sequence[float],
                 dropout_inplace: bool | Sequence[bool],
                 batch_norm: bool | Sequence[bool],
                 batch_norm_eps: float | Sequence[float],
                 batch_norm_momentum: float | Sequence[float],
                 batch_norm_affine: bool | Sequence[bool],
                 batch_norm_track_running_stats: bool | Sequence[bool]) -> None:
        """
        Initializes the ModuleMultiLayerConv1d.

        Args:
            num_layers (int): Number of layers.
            channel_size (int | Sequence[int]): Number of channels in each layer.
            kernel_size (int | Sequence[int]): Kernel size for each layer.
            stride (int | Sequence[int]): Stride for each layer.
            padding (Literal['same', 'valid'] | int | _size_1_t | Sequence[Literal['same', 'valid'] | int | _size_1_t]): Padding for each layer.
            dilation (int | _size_1_t | Sequence[int | _size_1_t]): Dilation for each layer.
            groups (int | Sequence[int]): Number of groups for each layer.
            bias (bool | Sequence[bool]): Whether to use bias in each layer.
            padding_mode (Literal['zeros', 'reflect', 'replicate', 'circular'] | Sequence[Literal['zeros', 'reflect', 'replicate', 'circular']]): Padding mode for each layer.
            dropout_p (float | Sequence[float]): Dropout probability for each layer.
            dropout_inplace (bool | Sequence[bool]): Whether to perform dropout in-place.
            batch_norm (bool | Sequence[bool]): Whether to apply batch normalization.
            batch_norm_eps (float | Sequence[float]): Epsilon value for batch normalization.
            batch_norm_momentum (float | Sequence[float]): Momentum value for batch normalization.
            batch_norm_affine (bool | Sequence[bool]): Whether to learn affine parameters in batch normalization.
            batch_norm_track_running_stats (bool | Sequence[bool]): Whether to track running statistics in batch normalization.
        """

        super(ModuleMultiLayerConv1d, self).__init__()

        self.num_batch_norm_layers = num_layers
        self.num_conv_layers = num_layers
        self.num_dropout_layers = num_layers - 1

        if isinstance(batch_norm, bool):
            batch_norm = [batch_norm] * num_layers
        if isinstance(batch_norm_eps, float):
            batch_norm_eps = [batch_norm_eps] * num_layers
        if isinstance(batch_norm_momentum, float):
            batch_norm_momentum = [batch_norm_momentum] * num_layers
        if isinstance(batch_norm_affine, bool):
            batch_norm_affine = [batch_norm_affine] * num_layers
        if isinstance(batch_norm_track_running_stats, bool):
            batch_norm_track_running_stats = [batch_norm_track_running_stats] * num_layers

        if isinstance(channel_size, int):
            channel_size = [channel_size] * (num_layers + 1)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        if isinstance(stride, int):
            stride = [stride] * num_layers
        if isinstance(padding, (int, str)):
            padding = [padding] * num_layers
        if isinstance(dilation, int):
            dilation = [dilation] * num_layers
        if isinstance(groups, int):
            groups = [groups] * num_layers
        if isinstance(bias, bool):
            bias = [bias] * num_layers
        if isinstance(padding_mode, str):
            padding_mode = [padding_mode] * num_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * (num_layers - 1)
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * (num_layers - 1)

        assert len(batch_norm) == num_layers, 'batch_norm must have length num_layers'
        assert len(batch_norm_eps) == num_layers, 'batch_norm_eps must have length num_layers'
        assert len(batch_norm_momentum) == num_layers, 'batch_norm_momentum must have length num_layers'
        assert len(batch_norm_affine) == num_layers, 'batch_norm_affine must have length num_layers'
        assert len(batch_norm_track_running_stats) == num_layers, 'batch_norm_track_running_stats must have length num_layers'

        assert len(channel_size) == num_layers + 1, 'channel_size must have length num_layers + 1'
        assert len(kernel_size) == num_layers, 'kernel_size must have length num_layers'
        assert len(stride) == num_layers, 'stride must have length num_layers'
        assert len(padding) == num_layers, 'padding must have length num_layers'
        assert len(dilation) == num_layers, 'dilation must have length num_layers'
        assert len(groups) == num_layers, 'groups must have length num_layers'
        assert len(bias) == num_layers, 'bias must have length num_layers'
        assert len(padding_mode) == num_layers, 'padding_mode must have length num_layers'

        assert len(dropout_p) == num_layers - 1, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == num_layers - 1, 'dropout_inplace must have length num_layers - 1'

        self.batch_norm = batch_norm
        self.batch_norm_eps = batch_norm_eps
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_affine = batch_norm_affine
        self.batch_norm_track_running_stats = batch_norm_track_running_stats

        self.conv_channel_size = channel_size
        self.conv_kernel_size = kernel_size
        self.conv_stride = stride
        self.conv_padding = padding
        self.conv_dilation = dilation
        self.conv_groups = groups
        self.conv_bias = bias
        self.conv_padding_mode = padding_mode

        self.dropout_p = dropout_p
        self.dropout_inplace = dropout_inplace

        self.modules = nn.Sequential()

        for i in range(num_layers):
            if batch_norm[i]:
                self.modules.append(nn.BatchNorm1d(num_features=channel_size[i],
                                                   eps=batch_norm_eps[i],
                                                   momentum=batch_norm_momentum[i],
                                                   affine=batch_norm_affine[i],
                                                   track_running_stats=batch_norm_track_running_stats[i]))
            self.modules.append(nn.Conv1d(in_channels=channel_size[i],
                                          out_channels=channel_size[i + 1],
                                          kernel_size=kernel_size[i],
                                          stride=stride[i],
                                          padding=padding[i],
                                          dilation=dilation[i],
                                          groups=groups[i],
                                          bias=bias[i]))
            if i < self.num_dropout_layers and dropout_p[i] > 0:
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
