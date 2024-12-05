from typing import Sequence, Self, Optional

import torch
from torch import nn, Tensor
from torch._prims_common import DeviceLikeType

from utils_torch.modules.LSTMLayerHiddenCell import LSTMLayerHiddenCell

hx_type = tuple[tuple[Tensor, ...], tuple[Tensor, ...]]

y_type = tuple[tuple[tuple[Tensor, ...], tuple[Tensor, ...]], tuple[tuple[Tensor, ...], tuple[Tensor, ...]]]


class MultiLayerHiddenCellLSTM(nn.Module):
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
    _layer_norm_elementwise_affine: tuple[bool, ...]
    _layer_norm_bias: tuple[bool, ...]

    _lstm_modules: nn.ModuleList
    _dropout_modules: nn.ModuleList
    _layer_norm_modules: nn.ModuleList

    _device: torch.device
    _dtype: torch.dtype

    _x_batch_dim: int
    _x_seq_dim: int
    _x_feature_dim: int
    _hx_direction_dim: int
    _hx_batch_dim: int
    _hx_hidden_dim: int
    _y_batch_dim: int
    _y_seq_dim: int
    _y_feature_dim: int
    _hy_direction_dim: int
    _hy_batch_dim: int
    _hy_hidden_dim: int

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
                 dropout_first: bool | Sequence[bool] = False,
                 layer_norm: bool | Sequence[bool] = False,
                 layer_norm_eps: float | Sequence[float] = 1e-5,
                 layer_norm_elementwise_affine: bool | Sequence[bool] = True,
                 layer_norm_bias: bool | Sequence[bool] = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        super(MultiLayerHiddenCellLSTM, self).__init__()

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
        if isinstance(layer_norm_elementwise_affine, bool):
            layer_norm_elementwise_affine = [layer_norm_elementwise_affine] * self._inner_num_layers
        if isinstance(layer_norm_bias, bool):
            layer_norm_bias = [layer_norm_bias] * self._inner_num_layers

        assert len(hidden_size) == self._lstm_num_layers, 'hidden_size must have length num_layers'
        assert len(proj_size) == self._lstm_num_layers, 'proj_size must have length num_layers'
        assert len(bidirectional) == self._lstm_num_layers, 'bidirectional must have length num_layers'
        assert len(bias) == self._lstm_num_layers, 'bias must have length num_layers'

        assert len(dropout_p) == self._inner_num_layers, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == self._inner_num_layers, 'dropout_inplace must have length num_layers - 1'
        assert len(dropout_first) == self._inner_num_layers, 'dropout_first must have length num_layers - 1'

        assert len(layer_norm) == self._inner_num_layers, 'layer_norm must have length num_layers - 1'
        assert len(layer_norm_eps) == self._inner_num_layers, 'layer_norm_eps must have length num_layers - 1'
        assert len(
            layer_norm_elementwise_affine) == self._inner_num_layers, 'layer_norm_elementwise_affine must have length num_layers - 1'
        assert len(layer_norm_bias) == self._inner_num_layers, 'layer_norm_bias must have length num_layers - 1'

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
        self._lstm_num_directions = tuple(2 if bidirectional_i else 1 for bidirectional_i in bidirectional)
        self._lstm_bias = tuple(bias)
        self._lstm_batch_first = batch_first

        self._dropout_p = tuple(dropout_p)
        self._dropout_inplace = tuple(dropout_inplace)
        self._dropout_first = tuple(dropout_first)

        self._layer_norm = tuple(layer_norm)
        self._layer_norm_eps = tuple(layer_norm_eps)
        self._layer_norm_elementwise_affine = tuple(layer_norm_elementwise_affine)
        self._layer_norm_bias = tuple(layer_norm_bias)

        self._lstm_modules = nn.ModuleList()
        self._dropout_modules = nn.ModuleList()
        self._layer_norm_modules = nn.ModuleList()

        self._device = device
        self._dtype = dtype

        self._x_batch_dim = 0 if self._lstm_batch_first else 1
        self._x_seq_dim = 1 if self._lstm_batch_first else 0
        self._x_feature_dim = 2
        self._hx_direction_dim = 0
        self._hx_batch_dim = 1
        self._hx_hidden_dim = 2
        self._y_batch_dim = 0 if self._lstm_batch_first else 1
        self._y_seq_dim = 1 if self._lstm_batch_first else 0
        self._y_feature_dim = 2
        self._hy_direction_dim = 0
        self._hy_batch_dim = 1
        self._hy_hidden_dim = 2

        for i in range(self._lstm_num_layers):
            self._lstm_modules.append(LSTMLayerHiddenCell(input_size=self._lstm_input_size[i],
                                                          hidden_size=self._lstm_hidden_size[i],
                                                          bidirectional=self._lstm_bidirectional[i],
                                                          proj_size=self._lstm_proj_size[i],
                                                          bias=self._lstm_bias[i],
                                                          batch_first=self._lstm_batch_first,
                                                          device=self._device,
                                                          dtype=self._dtype))
            if i >= self._inner_num_layers:
                continue
            if self._dropout_p[i] > 0.0:
                self._dropout_modules.append(nn.Dropout(p=self._dropout_p[i],
                                                        inplace=self._dropout_inplace[i]))
            else:
                self._dropout_modules.append(nn.Identity())
            if layer_norm[i]:
                self._layer_norm_modules.append(nn.LayerNorm(normalized_shape=self._lstm_output_size[i],
                                                             eps=self._layer_norm_eps[i],
                                                             elementwise_affine=self._layer_norm_elementwise_affine[i],
                                                             bias=self._layer_norm_bias[i],
                                                             device=self._device,
                                                             dtype=self._dtype))
            else:
                self._layer_norm_modules.append(nn.Identity())

    def to(self,
           device: Optional[DeviceLikeType] = ...,
           dtype: Optional[torch.dtype] = ...,
           non_blocking: bool = ..., ) -> Self:
        if device is not ...:
            self._device = device
        if dtype is not ...:
            self._dtype = dtype
        return super().to(device, dtype, non_blocking)

    def forward(self,
                x: Tensor,
                hx: hx_type = None) -> y_type:

        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(self._x_batch_dim)
        self.check_x(x)

        batch_size = x.size(self._x_batch_dim)

        if hx is None:
            hx = self.get_hx(batch_size, self._device, self._dtype)
        h_0, c_0 = hx
        if not is_batched:
            h_0 = tuple(h_0_i.unsqueeze(self._hx_batch_dim) for h_0_i in h_0)
            c_0 = tuple(c_0_i.unsqueeze(self._hx_batch_dim) for c_0_i in c_0)
        self.check_h_0(h_0, batch_size)
        self.check_c_0(c_0, batch_size)

        x_i = x
        y_hidden = []
        y_cell = []
        last_hidden = []
        last_cell = []
        for i in range(self._lstm_num_layers):
            h_0_i = h_0[i].contiguous()
            c_0_i = c_0[i].contiguous()
            (y_hidden_i, y_cell_i), (last_hidden_i, last_cell_i) = self._lstm_modules[i](x_i, (h_0_i, c_0_i))
            y_hidden.append(y_hidden_i)
            y_cell.append(y_cell_i)
            last_hidden.append(last_hidden_i)
            last_cell.append(last_cell_i)
            if i < self._inner_num_layers:
                x_i = self._dropout_modules[i](y_hidden_i)
                x_i = self._layer_norm_modules[i](x_i)
            else:
                x_i = y_hidden_i

        y_hidden = tuple(y_hidden)
        y_cell = tuple(y_cell)
        last_hidden = tuple(last_hidden)
        last_cell = tuple(last_cell)
        return (y_hidden, y_cell), (last_hidden, last_cell)

    def get_hx_size(self,
                    batch_size: int) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
        h_0_size = []
        c_0_size = []
        for i in range(self._lstm_num_layers):
            h_0_size_i = (self._lstm_num_directions[i],
                          batch_size,
                          self._lstm_hidden_size[i] if self._lstm_proj_size[i] == 0 else self._lstm_proj_size[i])
            c_0_size_i = (self._lstm_num_directions[i],
                          batch_size,
                          self._lstm_hidden_size[i])
            h_0_size.append(h_0_size_i)
            c_0_size.append(c_0_size_i)
        return tuple(h_0_size), tuple(c_0_size)

    def get_hx(self,
               batch_size: int,
               device: torch.device = None,
               dtype: torch.dtype = None) -> hx_type:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        hx_size = self.get_hx_size(batch_size)
        h_0 = [torch.zeros(h_0_size_i, device=device, dtype=dtype) for h_0_size_i in hx_size[0]]
        c_0 = [torch.zeros(c_0_size_i, device=device, dtype=dtype) for c_0_size_i in hx_size[1]]
        return tuple(h_0), tuple(c_0)

    def check_h_0(self,
                  h_0: tuple[Tensor, ...],
                  batch_size: int) -> None:
        for i in range(self._lstm_num_layers):
            h_0_i = h_0[i]
            if h_0_i.dim() != 3:
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected h_0 to be 3D, got {h_0_i.dim()}D instead")
            if h_0_i.size(self._hx_direction_dim) != self._lstm_num_directions[i]:
                raise ValueError(
                    f"MultiLayerHiddenCellLSTM: Expected h_0 to have size {self._lstm_num_directions[i]} in the direction dimension, got size {h_0_i.size(self._hx_direction_dim)} instead")
            if h_0_i.size(self._hx_batch_dim) != batch_size:
                raise ValueError(
                    f"MultiLayerHiddenCellLSTM: Expected h_0 to have size {batch_size} in the batch dimension, got size {h_0_i.size(self._hx_batch_dim)} instead")
            if h_0_i.size(self._hx_hidden_dim) != (
                    self._lstm_proj_size[i] if self._lstm_proj_size[i] > 0 else self._lstm_hidden_size[i]):
                raise ValueError(
                    f"MultiLayerHiddenCellLSTM: Expected h_0 to have size {self._lstm_output_size[i]} in the hidden dimension, got size {h_0_i.size(self._hx_hidden_dim)} instead")

    def check_c_0(self,
                  c_0: tuple[Tensor, ...],
                  batch_size: int) -> None:
        for i in range(self._lstm_num_layers):
            c_0_i = c_0[i]
            if c_0_i.dim() != 3:
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected c_0 to be 3D, got {c_0_i.dim()}D instead")
            if c_0_i.size(self._hx_direction_dim) != self._lstm_num_directions[i]:
                raise ValueError(
                    f"MultiLayerHiddenCellLSTM: Expected c_0 to have size {self._lstm_num_directions[i]} in the direction dimension, got size {c_0_i.size(self._hx_direction_dim)} instead")
            if c_0_i.size(self._hx_batch_dim) != batch_size:
                raise ValueError(
                    f"MultiLayerHiddenCellLSTM: Expected c_0 to have size {batch_size} in the batch dimension, got size {c_0_i.size(self._hx_batch_dim)} instead")
            if c_0_i.size(self._hx_hidden_dim) != self._lstm_hidden_size[i]:
                raise ValueError(
                    f"MultiLayerHiddenCellLSTM: Expected c_0 to have size {self._lstm_hidden_size[i]} in the hidden dimension, got size {c_0_i.size(self._hx_hidden_dim)} instead")

    def check_x(self, x: Tensor) -> None:
        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"MultiLayerHiddenCellLSTM: Expected x to be 2D or 3D, got {x.dim()}D instead")
        if x.size(self._x_feature_dim) != self._lstm_input_size[0]:
            raise ValueError(
                f"MultiLayerHiddenCellLSTM: Expected x to have size {self._lstm_input_size[0]} in the feature dimension, got size {x.size(self._x_feature_dim)} instead")

    @property
    def lstm_num_layers(self) -> int:
        return self._lstm_num_layers

    @property
    def inner_num_layers(self) -> int:
        return self._inner_num_layers

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
    def lstm_layer_norm_elementwise_affine(self) -> tuple[bool, ...]:
        return self._layer_norm_elementwise_affine

    @property
    def lstm_layer_norm_bias(self) -> tuple[bool, ...]:
        return self._layer_norm_bias
