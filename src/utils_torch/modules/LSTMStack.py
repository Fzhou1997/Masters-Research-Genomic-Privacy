from typing import Sequence

import torch
from torch import nn, Tensor

from utils_torch.modules import LSTMLayer


class LSTMStack(nn.Module):
    _lstm_num_layers: int
    _inner_num_layers: int

    _lstm_input_size: tuple[int, ...]
    _lstm_hidden_size: tuple[int, ...]
    _lstm_proj_size: tuple[int, ...]
    _lstm_output_size: tuple[int, ...]
    _lstm_bidirectional: tuple[bool, ...]
    _lstm_num_directions: tuple[int, ...]
    _lstm_bias: tuple[bool, ...]
    _lstm_batch_first: bool

    _dropout_p: tuple[float, ...]
    _dropout_inplace: tuple[bool, ...]
    _dropout_first: tuple[bool, ...]

    _layer_norm: tuple[bool, ...]
    _layer_norm_eps: tuple[float, ...]
    _layer_norm_element_wise_affine: tuple[bool, ...]
    _layer_norm_bias: tuple[bool, ...]

    _lstm_modules: nn.ModuleList
    _dropout_modules: nn.ModuleList
    _layer_norm_modules: nn.ModuleList

    def __init__(self,
                 num_layers: int,
                 input_size: int,
                 hidden_size: int | Sequence[int],
                 proj_size: int | Sequence[int] = 0,
                 bidirectional: bool | Sequence[bool] = False,
                 bias: bool | Sequence[bool] = True,
                 batch_first: bool = True,
                 dropout_p: float | Sequence[float] = 0.5,
                 dropout_inplace: bool | Sequence[bool] = False,
                 dropout_first: bool | Sequence[bool] = True,
                 layer_norm: bool | Sequence[bool] = True,
                 layer_norm_eps: float | Sequence[float] = 1e-5,
                 layer_norm_element_wise_affine: bool | Sequence[bool] = True,
                 layer_norm_bias: bool | Sequence[bool] = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        super(LSTMStack, self).__init__()

        assert num_layers > 0, 'num_layers must be greater than 0'

        self._lstm_num_layers = num_layers
        self._inner_num_layers = num_layers - 1

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * self._lstm_num_layers
        if isinstance(proj_size, int):
            proj_size = [proj_size] * self._lstm_num_layers
        if isinstance(bidirectional, bool):
            bidirectional = [bidirectional] * self._lstm_num_layers
        if isinstance(bias, bool):
            bias = [bias] * self._lstm_num_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * self._inner_num_layers
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * self._inner_num_layers
        if isinstance(dropout_first, bool):
            dropout_first = [dropout_first] * self._inner_num_layers

        if isinstance(layer_norm, bool):
            layer_norm = [layer_norm] * self._inner_num_layers
        if isinstance(layer_norm_eps, float):
            layer_norm_eps = [layer_norm_eps] * self._inner_num_layers
        if isinstance(layer_norm_element_wise_affine, bool):
            layer_norm_element_wise_affine = [
                                                 layer_norm_element_wise_affine] * self._inner_num_layers
        if isinstance(layer_norm_bias, bool):
            layer_norm_bias = [layer_norm_bias] * self._inner_num_layers

        assert len(hidden_size) == self._lstm_num_layers, 'hidden_size must have length num_layers'
        assert len(proj_size) == self._lstm_num_layers, 'proj_size must have length num_layers'
        assert len(
            bidirectional) == self._lstm_num_layers, 'bidirectional must have length num_layers'
        assert len(bias) == self._lstm_num_layers, 'bias must have length num_layers'

        assert len(dropout_p) == self._inner_num_layers, 'dropout_p must have length num_layers - 1'
        assert len(
            dropout_inplace) == self._inner_num_layers, 'dropout_inplace must have length num_layers - 1'
        assert len(
            dropout_first) == self._inner_num_layers, 'dropout_first must have length num_layers - 1'

        assert len(
            layer_norm) == self._inner_num_layers, 'layer_norm must have length num_layers - 1'
        assert len(
            layer_norm_eps) == self._inner_num_layers, 'layer_norm_eps must have length num_layers - 1'
        assert len(
            layer_norm_element_wise_affine) == self._inner_num_layers, 'layer_norm_element_wise_affine must have length num_layers - 1'
        assert len(
            layer_norm_bias) == self._inner_num_layers, 'layer_norm_bias must have length num_layers - 1'

        output_size = []
        for i in range(self._lstm_num_layers):
            output_size_i = proj_size[i] if proj_size[i] > 0 else hidden_size[i]
            output_size_i *= 2 if bidirectional[i] else 1
            output_size.append(output_size_i)
        self._lstm_output_size = tuple(output_size)

        input_sizes = [input_size] + output_size[:-1]
        self._lstm_input_size = tuple(input_sizes)

        self._lstm_hidden_size = tuple(hidden_size)
        self._lstm_proj_size = tuple(proj_size)
        self._lstm_bidirectional = tuple(bidirectional)
        self._lstm_num_directions = tuple(
            [2 if bidirectional_i else 1 for bidirectional_i in bidirectional])
        self._lstm_bias = tuple(bias)
        self._lstm_batch_first = batch_first

        self._dropout_p = tuple(dropout_p)
        self._dropout_inplace = tuple(dropout_inplace)
        self._dropout_first = tuple(dropout_first)

        self._layer_norm = tuple(layer_norm)
        self._layer_norm_eps = tuple(layer_norm_eps)
        self._layer_norm_element_wise_affine = tuple(layer_norm_element_wise_affine)
        self._layer_norm_bias = tuple(layer_norm_bias)

        self._lstm_modules = nn.ModuleList()
        self._dropout_modules = nn.ModuleList()
        self._layer_norm_modules = nn.ModuleList()

        for i in range(num_layers):
            self._lstm_modules.append(LSTMLayer(input_size=input_sizes[i],
                                                hidden_size=hidden_size[i],
                                                proj_size=proj_size[i],
                                                bidirectional=bidirectional[i],
                                                bias=bias[i],
                                                batch_first=batch_first,
                                                device=device,
                                                dtype=dtype))
            if i >= self._inner_num_layers:
                continue
            if dropout_p[i] > 0.0:
                self._dropout_modules.append(nn.Dropout(p=dropout_p[i],
                                                        inplace=dropout_inplace[i]))
            else:
                self._dropout_modules.append(nn.Identity())
            if layer_norm[i]:
                self._layer_norm_modules.append(nn.LayerNorm(normalized_shape=self._lstm_output_size[i],
                                                             eps=layer_norm_eps[i],
                                                             elementwise_affine=
                                                             layer_norm_element_wise_affine[i],
                                                             bias=layer_norm_bias[i],
                                                             device=device,
                                                             dtype=dtype))
            else:
                self._layer_norm_modules.append(nn.Identity())

    @property
    def lstm_num_layers(self) -> int:
        return self._lstm_num_layers

    @property
    def lstm_input_size_in(self) -> int:
        return self._lstm_input_size[0]

    @property
    def lstm_output_size_out(self) -> int:
        return self._lstm_output_size[-1]

    @property
    def lstm_input_size(self) -> tuple[int, ...]:
        return self._lstm_input_size

    @property
    def lstm_hidden_size(self) -> tuple[int, ...]:
        return self._lstm_hidden_size

    @property
    def lstm_proj_size(self) -> tuple[int, ...]:
        return self._lstm_proj_size

    @property
    def lstm_output_size(self) -> tuple[int, ...]:
        return self._lstm_output_size

    @property
    def lstm_bidirectional(self) -> tuple[bool, ...]:
        return self._lstm_bidirectional

    @property
    def lstm_num_directions(self) -> tuple[int, ...]:
        return self._lstm_num_directions

    @property
    def lstm_bias(self) -> tuple[bool, ...]:
        return self._lstm_bias

    @property
    def lstm_batch_first(self) -> bool:
        return self._lstm_batch_first

    @property
    def lstm_dropout_p(self) -> tuple[float, ...]:
        return self._dropout_p

    @property
    def lstm_dropout_inplace(self) -> tuple[bool, ...]:
        return self._dropout_inplace

    @property
    def lstm_dropout_first(self) -> tuple[bool, ...]:
        return self._dropout_first

    @property
    def lstm_layer_norm(self) -> tuple[bool, ...]:
        return self._layer_norm

    @property
    def lstm_layer_norm_eps(self) -> tuple[float, ...]:
        return self._layer_norm_eps

    @property
    def lstm_layer_norm_element_wise_affine(self) -> tuple[bool, ...]:
        return self._layer_norm_element_wise_affine

    @property
    def lstm_layer_norm_bias(self) -> tuple[bool, ...]:
        return self._layer_norm_bias

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def lstm_forward_input_weights(self) -> tuple[Tensor, ...]:
        return tuple([lstm.forward_input_weights for lstm in self._lstm_modules])

    @property
    def lstm_forward_hidden_weights(self) -> tuple[Tensor, ...]:
        return tuple([lstm.forward_hidden_weights for lstm in self._lstm_modules])

    @property
    def lstm_forward_input_bias(self) -> tuple[Tensor, ...]:
        return tuple([lstm.forward_input_bias for lstm in self._lstm_modules])

    @property
    def lstm_forward_hidden_bias(self) -> tuple[Tensor, ...]:
        return tuple([lstm.forward_hidden_bias for lstm in self._lstm_modules])

    @property
    def lstm_forward_projection_weights(self) -> tuple[Tensor | None, ...]:
        return tuple([lstm.forward_projection_weights for lstm in self._lstm_modules])

    @property
    def lstm_backward_input_weights(self) -> tuple[Tensor | None, ...]:
        return tuple([lstm.backward_input_weights for lstm in self._lstm_modules])

    @property
    def lstm_backward_hidden_weights(self) -> tuple[Tensor | None, ...]:
        return tuple([lstm.backward_hidden_weights for lstm in self._lstm_modules])

    @property
    def lstm_backward_input_bias(self) -> tuple[Tensor | None, ...]:
        return tuple([lstm.backward_input_bias for lstm in self._lstm_modules])

    @property
    def lstm_backward_hidden_bias(self) -> tuple[Tensor | None, ...]:
        return tuple([lstm.backward_hidden_bias for lstm in self._lstm_modules])

    @property
    def lstm_backward_projection_weights(self) -> tuple[Tensor | None, ...]:
        return tuple([lstm.backward_projection_weights for lstm in self._lstm_modules])

    def get_parameters(self) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None], ...]:
        return tuple([lstm.get_parameters() for lstm in self._lstm_modules])

    def set_parameters(self, parameters: tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None], ...]) -> None:
        for i, lstm in enumerate(self._lstm_modules):
            lstm.set_parameters(parameters[i])

    def get_hx_size(self, batch_size: int) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
        return tuple([lstm.get_hx_size(batch_size=batch_size) for lstm in self._lstm_modules])

    def get_hx(self, batch_size: int) -> tuple[tuple[Tensor, Tensor], ...]:
        return tuple([lstm.get_hx(batch_size=batch_size) for lstm in self._lstm_modules])

    def check_hx(self, hx: tuple[tuple[Tensor, Tensor], ...], batch_size: int) -> None:
        for i, lstm in enumerate(self._lstm_modules):
            lstm.check_hx(hx[i], batch_size=batch_size)

    def forward(self, x: Tensor, hx: tuple[tuple[Tensor, Tensor], ...] = None) -> tuple[tuple[Tensor, tuple[Tensor, Tensor]], ...]:
        x_i = x
        if hx is None:
            hx = [None] * self._lstm_num_layers
        outputs = []
        for i, lstm in enumerate(self._lstm_modules):
            hx_i = hx[i]
            x_i, hx_i = lstm.forward(x_i, hx_i)
            outputs.append((x_i, hx_i))
            if i >= self._inner_num_layers:
                continue
            if self._dropout_first[i]:
                x_i = self._dropout_modules[i](x_i)
                x_i = self._layer_norm_modules[i](x_i)
            else:
                x_i = self._layer_norm_modules[i](x_i)
                x_i = self._dropout_modules[i](x_i)
        return tuple(outputs)

