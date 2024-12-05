from typing import Literal, Type, Sequence

import torch
from torch import nn, Tensor

_activations = {
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
    "LogSoftmax"
}

class Conv1dStack(nn.Module):
    _conv_num_layers: int
    _inner_num_layers: int

    _conv_channel_size: tuple[int, ...]
    _conv_kernel_size: tuple[int, ...]
    _conv_stride: tuple[int, ...]
    _conv_padding: tuple[Literal['same', 'valid'] | int | _size_1_t, ...]
    _conv_dilation: tuple[int, ...]
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

    _conv_modules: nn.ModuleList
    _activation_modules: nn.ModuleList
    _dropout_modules: nn.ModuleList
    _batch_norm_modules: nn.ModuleList

    def __init__(self,
                 num_layers: int,
                 channel_size: int | Sequence[int],
                 kernel_size: int | Sequence[int],
                 stride: int | Sequence[int] = 1,
                 padding: Literal['same', 'valid'] | int | Sequence[
                     Literal['same', 'valid'] | int] = 0,
                 dilation: int | Sequence[int] = 1,
                 groups: int | Sequence[int] = 1,
                 bias: bool | Sequence[bool] = True,
                 padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] | Sequence[
                     Literal['zeros', 'reflect', 'replicate', 'circular']] = 'zeros',
                 activation: Type[nn.Module] | Sequence[Type[nn.Module]] = nn.ReLU,
                 activation_kwargs: dict[str, any] | Sequence[dict[str, any]] = None,
                 dropout_p: float | Sequence[float] = 0.5,
                 dropout_inplace: bool | Sequence[bool] = False,
                 dropout_first: bool | Sequence[bool] = True,
                 batch_norm: bool | Sequence[bool] = True,
                 batch_norm_eps: float | Sequence[float] = 1e-5,
                 batch_norm_momentum: float | Sequence[float] = 0.1,
                 batch_norm_affine: bool | Sequence[bool] = True,
                 batch_norm_track_running_stats: bool | Sequence[bool] = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        super(Conv1dStack, self).__init__()

        assert num_layers > 0, "num_layers must be greater than 0"

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
            batch_norm_track_running_stats = [
                                                 batch_norm_track_running_stats] * self._inner_num_layers

        assert len(channel_size) == num_layers + 1, 'channel_size must have length num_layers + 1'
        assert len(kernel_size) == num_layers, 'kernel_size must have length num_layers'
        assert len(stride) == num_layers, 'stride must have length num_layers'
        assert len(padding) == num_layers, 'padding must have length num_layers'
        assert len(dilation) == num_layers, 'dilation must have length num_layers'
        assert len(groups) == num_layers, 'groups must have length num_layers'
        assert len(bias) == num_layers, 'bias must have length num_layers'
        assert len(padding_mode) == num_layers, 'padding_mode must have length num_layers'

        assert len(activation) == num_layers - 1, 'activation must have length num_layers - 1'
        assert len(
            activation_kwargs) == num_layers - 1, 'activation_kwargs must have length num_layers - 1'
        assert all([cls is None or cls.__name__ in _activations for cls in
                    activation]), 'activation must be one of the following types: ' + ', '.join(
            _activations)

        assert len(dropout_p) == num_layers - 1, 'dropout_p must have length num_layers - 1'
        assert len(
            dropout_inplace) == num_layers - 1, 'dropout_inplace must have length num_layers - 1'
        assert len(dropout_first) == num_layers - 1, 'dropout_first must have length num_layers - 1'

        assert len(batch_norm) == num_layers - 1, 'batch_norm must have length num_layers - 1'
        assert len(
            batch_norm_eps) == num_layers - 1, 'batch_norm_eps must have length num_layers - 1'
        assert len(
            batch_norm_momentum) == num_layers - 1, 'batch_norm_momentum must have length num_layers - 1'
        assert len(
            batch_norm_affine) == num_layers - 1, 'batch_norm_affine must have length num_layers - 1'
        assert len(
            batch_norm_track_running_stats) == num_layers - 1, 'batch_norm_track_running_stats must have length num_layers - 1'

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

        self._conv_modules = nn.ModuleList()
        self._activation_modules = nn.ModuleList()
        self._dropout_modules = nn.ModuleList()
        self._batch_norm_modules = nn.ModuleList()

        for i in range(self._conv_num_layers):
            self._conv_modules.append(nn.Conv1d(in_channels=self._conv_channel_size[i],
                                                out_channels=self._conv_channel_size[i + 1],
                                                kernel_size=self._conv_kernel_size[i],
                                                stride=self._conv_stride[i],
                                                padding=self._conv_padding[i],
                                                dilation=self._conv_dilation[i],
                                                groups=self._conv_groups[i],
                                                bias=self._conv_bias[i],
                                                padding_mode=self._conv_padding_mode[i],
                                                device=device,
                                                dtype=dtype))
            if i >= self._inner_num_layers:
                continue
            self._activation_modules.append(activation[i](**activation_kwargs[i]))
            if dropout_p[i] > 0:
                self._dropout_modules.append(nn.Dropout(p=self._dropout_p[i],
                                                        inplace=self._dropout_inplace[i]))
            else:
                self._dropout_modules.append(nn.Identity())
            if batch_norm[i]:
                self._batch_norm_modules.append(
                    nn.BatchNorm1d(num_features=self._conv_channel_size[i + 1],
                                   eps=self._batch_norm_eps[i],
                                   momentum=self._batch_norm_momentum[i],
                                   affine=self._batch_norm_affine[i],
                                   track_running_stats=self._batch_norm_track_running_stats[i],
                                   device=device,
                                   dtype=dtype))
            else:
                self._batch_norm_modules.append(nn.Identity())

    @property
    def conv_num_layers(self) -> int:
        return self._conv_num_layers

    @property
    def conv_channel_size_in(self) -> int:
        return self._conv_channel_size[0]

    @property
    def conv_channel_size_out(self) -> int:
        return self._conv_channel_size[-1]

    @property
    def conv_channel_size(self) -> tuple[int, ...]:
        return self._conv_channel_size

    @property
    def conv_kernel_size(self) -> tuple[int, ...]:
        return self._conv_kernel_size

    @property
    def conv_stride(self) -> tuple[int, ...]:
        return self._conv_stride

    @property
    def conv_padding(self) -> tuple[Literal['same', 'valid'] | int, ...]:
        return self._conv_padding

    @property
    def conv_dilation(self) -> tuple[int, ...]:
        return self._conv_dilation

    @property
    def conv_groups(self) -> tuple[int, ...]:
        return self._conv_groups

    @property
    def conv_bias(self) -> tuple[bool, ...]:
        return self._conv_bias

    @property
    def conv_padding_mode(self) -> tuple[Literal['zeros', 'reflect', 'replicate', 'circular'], ...]:
        return self._conv_padding_mode

    @property
    def conv_activation(self) -> tuple[Type[nn.Module], ...]:
        return self._activation

    @property
    def conv_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        return self._activation_kwargs

    @property
    def conv_dropout_p(self) -> tuple[float, ...]:
        return self._dropout_p

    @property
    def conv_dropout_inplace(self) -> tuple[bool, ...]:
        return self._dropout_inplace

    @property
    def conv_dropout_first(self) -> tuple[bool, ...]:
        return self._dropout_first

    @property
    def conv_batch_norm(self) -> tuple[bool, ...]:
        return self._batch_norm

    @property
    def conv_batch_norm_eps(self) -> tuple[float, ...]:
        return self._batch_norm_eps

    @property
    def conv_batch_norm_momentum(self) -> tuple[float, ...]:
        return self._batch_norm_momentum

    @property
    def conv_batch_norm_affine(self) -> tuple[bool, ...]:
        return self._batch_norm_affine

    @property
    def conv_batch_norm_track_running_stats(self) -> tuple[bool, ...]:
        return self._batch_norm_track_running_stats

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_seq_size_out(self, seq_size_in: int) -> int:
        seq_size_out = seq_size_in
        for i in range(self.conv_num_layers):
            seq_size_out = (seq_size_out + 2 * self.conv_padding[i] - self.conv_dilation[i] * (self.conv_kernel_size[i] - 1) - 1) // self.conv_stride[i] + 1
        return seq_size_out

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.conv_num_layers):
            x = self._conv_modules[i](x)
            if i >= self._inner_num_layers:
                continue
            x = self._activation_modules[i](x)
            if self._dropout_first[i]:
                x = self._dropout_modules[i](x)
                x = self._batch_norm_modules[i](x)
            else:
                x = self._batch_norm_modules[i](x)
                x = self._dropout_modules[i](x)
        return x