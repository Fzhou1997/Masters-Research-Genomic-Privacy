from typing import Sequence, Self, Optional

import torch
from torch import nn, Tensor
from torch._prims_common import DeviceLikeType

from utils_torch.modules.HiddenCellLSTM import HiddenCellLSTM


hx_type = tuple[tuple[Tensor, ...], tuple[Tensor, ...]]

y_type = tuple[
    tuple[
        tuple[Tensor, ...],
        tuple[Tensor, ...]
    ],
    tuple[
        tuple[Tensor, ...],
        tuple[Tensor, ...]
    ]
]

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

    def __init__(self,
                 num_layers: int,
                 input_size: int,
                 hidden_size: int | Sequence[int],
                 proj_size: int | Sequence[int] = 0,
                 bidirectional: bool | Sequence[bool] = False,
                 bias: bool | Sequence[bool] = True,
                 batch_first: bool | Sequence[bool] = False,
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

        for i in range(self._lstm_num_layers):
            self._lstm_modules.append(HiddenCellLSTM(input_size=self._lstm_input_size[i],
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
           non_blocking: bool = ...,) -> Self:
        if device is not ...:
            self._device = device
        if dtype is not ...:
            self._dtype = dtype
        return super().to(device, dtype, non_blocking)

    def forward(self,
                x: Tensor,
                hx: hx_type = None) -> tuple[Tensor, hx_type]:
        pass #TODO: implement

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

    def check_hx(self, hx: hx_type) -> None:
        batch_size = hx[0][0].size(1)
        hx_size = self.get_hx_size(batch_size)
        if len(hx) != len(hx_size):
            raise ValueError(f"MultiLayerHiddenCellLSTM: Expected hx to be a tuple of length {len(hx_size)}, got length {len(hx)} instead")
        if len(hx[0]) != len(hx_size[0]):
            raise ValueError(f"MultiLayerHiddenCellLSTM: Expected h_0 to be a tuple of length {len(hx_size[0])}, got length {len(hx[0])} instead")
        if len(hx[1]) != len(hx_size[1]):
            raise ValueError(f"MultiLayerHiddenCellLSTM: Expected c_0 to be a tuple of length {len(hx_size[1])}, got length {len(hx[1])} instead")
        for i in range(self._lstm_num_layers):
            h_0_i = hx[0][i]
            c_0_i = hx[1][i]
            h_0_i_size = hx_size[0][i]
            c_0_i_size = hx_size[1][i]
            if h_0_i.dim() != len(h_0_i_size):
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected h_0 to be {len(h_0_i_size)}D, got {h_0_i.dim()}D instead")
            if c_0_i.dim() != len(c_0_i_size):
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected c_0 to be {len(c_0_i_size)}D, got {c_0_i.dim()}D instead")
            if h_0_i.size() != h_0_i_size:
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected h_0 to have size {h_0_i_size}, got size {h_0_i.size()} instead")
            if c_0_i.size() != c_0_i_size:
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected c_0 to have size {c_0_i_size}, got size {c_0_i.size()} instead")
            if h_0_i.size(1) != batch_size:
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected h_0 to have size {batch_size} in the batch dimension, got size {h_0_i.size(1)} instead")
            if c_0_i.size(1) != batch_size:
                raise ValueError(f"MultiLayerHiddenCellLSTM: Expected c_0 to have size {batch_size} in the batch dimension, got size {c_0_i.size(1)} instead")