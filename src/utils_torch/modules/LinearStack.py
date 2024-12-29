from typing import Sequence, Type

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
    "LogSoftmax",
}


class LinearStack(nn.Module):

    _linear_num_layers: int
    _inner_num_layers: int

    _linear_num_features: tuple[int, ...]
    _linear_bias: tuple[bool, ...]

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

    _linear_modules: nn.ModuleList
    _activation_modules: nn.ModuleList
    _dropout_modules: nn.ModuleList
    _batch_norm_modules: nn.ModuleList

    def __init__(self,
                 num_layers: int,
                 num_features: int | Sequence[int],
                 bias: bool | Sequence[bool] = True,
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
        super(LinearStack, self).__init__()

        assert num_layers > 0, f"num_layers must be greater than 0"

        self._linear_num_layers = num_layers
        self._inner_num_layers = num_layers - 1

        if isinstance(num_features, int):
            num_features = [num_features] * (self._linear_num_layers + 1)
        if isinstance(bias, bool):
            bias = [bias] * self._linear_num_layers

        if isinstance(activation, type):
            activation = [activation] * self._inner_num_layers
        if isinstance(activation_kwargs, dict):
            activation_kwargs = [activation_kwargs] * self._inner_num_layers
        if activation is None:
            activation = [nn.Identity] * self._inner_num_layers
        if activation_kwargs is None:
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

        assert len(num_features) == num_layers + 1, 'num_features must have length num_layers + 1'
        assert len(bias) == num_layers, 'bias must have length num_layers'

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

        self._linear_num_features = tuple(num_features)
        self._linear_bias = tuple(bias)

        for i in range(self._inner_num_layers):
            if activation[i] is None:
                activation[i] = nn.Identity
            if activation_kwargs[i] is None:
                activation_kwargs[i] = {}
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

        self._linear_modules = nn.ModuleList()
        self._activation_modules = nn.ModuleList()
        self._dropout_modules = nn.ModuleList()
        self._batch_norm_modules = nn.ModuleList()

        for i in range(self._linear_num_layers):
            self._linear_modules.append(nn.Linear(in_features=self._linear_num_features[i],
                                                  out_features=self._linear_num_features[i + 1],
                                                  bias=self._linear_bias[i],
                                                  device=device,
                                                  dtype=dtype))
            if i >= self._inner_num_layers:
                continue
            self._activation_modules.append(self._activation[i](**self._activation_kwargs[i]))
            if self._dropout_p[i] > 0:
                self._dropout_modules.append(nn.Dropout(p=self._dropout_p[i],
                                                        inplace=self._dropout_inplace[i]))
            else:
                self._dropout_modules.append(nn.Identity())
            if self._batch_norm[i]:
                self._batch_norm_modules.append(nn.BatchNorm1d(num_features=self._linear_num_features[i + 1],
                                                               eps=self._batch_norm_eps[i],
                                                               momentum=self._batch_norm_momentum[i],
                                                               affine=self._batch_norm_affine[i],
                                                               track_running_stats=self._batch_norm_track_running_stats[i],
                                                               device=device,
                                                               dtype=dtype))
            else:
                self._batch_norm_modules.append(nn.Identity())


    @property
    def linear_num_layers(self) -> int:
        return self._linear_num_layers

    @property
    def linear_num_features_in(self) -> int:
        return self._linear_num_features[0]

    @property
    def linear_num_features_out(self) -> int:
        return self._linear_num_features[-1]

    @property
    def linear_num_features(self) -> tuple[int, ...]:
        return self._linear_num_features

    @property
    def linear_bias(self) -> tuple[bool, ...]:
        return self._linear_bias

    @property
    def linear_activation(self) -> tuple[Type[nn.Module], ...]:
        return self._activation

    @property
    def linear_activation_kwargs(self) -> tuple[dict[str, any], ...]:
        return self._activation_kwargs

    @property
    def linear_dropout_p(self) -> tuple[float, ...]:
        return self._dropout_p

    @property
    def linear_dropout_inplace(self) -> tuple[bool, ...]:
        return self._dropout_inplace

    @property
    def linear_dropout_first(self) -> tuple[bool, ...]:
        return self._dropout_first

    @property
    def linear_batch_norm(self) -> tuple[bool, ...]:
        return self._batch_norm

    @property
    def linear_batch_norm_eps(self) -> tuple[float, ...]:
        return self._batch_norm_eps

    @property
    def linear_batch_norm_momentum(self) -> tuple[float, ...]:
        return self._batch_norm_momentum

    @property
    def linear_batch_norm_affine(self) -> tuple[bool, ...]:
        return self._batch_norm_affine

    @property
    def linear_batch_norm_track_running_stats(self) -> tuple[bool, ...]:
        return self._batch_norm_track_running_stats

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self._linear_num_layers):
            x = self._linear_modules[i](x)
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