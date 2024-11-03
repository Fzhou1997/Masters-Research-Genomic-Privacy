from typing import Sequence, Union

import torch
from torch import nn, Size, Tensor


_shape_t = Union[int, list[int], Size]

class MultiLayerLSTM(nn.Module):

    num_lstm_layers: int
    num_inner_layers: int

    lstm_feature_sizes: Sequence[int]
    lstm_bias: Sequence[bool]
    lstm_batch_first: bool
    lstm_bidirectional: Sequence[bool]
    lstm_proj_size: Sequence[int]

    dropout_p: Sequence[float]
    dropout_inplace: Sequence[bool]
    dropout_first: Sequence[bool]

    layer_norm: Sequence[bool]
    layer_norm_eps: Sequence[float]
    layer_norm_element_wise_affine: Sequence[bool]
    layer_norm_bias: Sequence[bool]

    lstm_modules: nn.ModuleList
    dropout_modules: nn.ModuleList
    layer_norm_modules: nn.ModuleList

    def __init__(self,
                 num_layers: int,
                 intput_size: int,
                 hidden_size: int | Sequence[int],
                 feature_size: int | Sequence[int],
                 proj_size: int | Sequence[int] = 0,
                 bidirectional: bool | Sequence[bool] = False,
                 bias: bool | Sequence[bool] = True,
                 batch_first: bool = True,
                 dropout_p: float | Sequence[float] = 0.5,
                 dropout_inplace: bool | Sequence[bool] = True,
                 dropout_first: bool | Sequence[bool] = True,
                 layer_norm: bool | Sequence[bool] = True,
                 layer_norm_eps: float | Sequence[float] = 1e-5,
                 layer_norm_element_wise_affine: bool | Sequence[bool] = True,
                 layer_norm_bias: bool | Sequence[bool] = True) -> None:

        super(MultiLayerLSTM, self).__init__()

        self.num_lstm_layers = num_layers
        self.num_inner_layers = num_layers - 1

        if isinstance(feature_size, int):
            feature_size = [feature_size] * (self.num_lstm_layers + 1)
        if isinstance(bias, bool):
            bias = [bias] * self.num_lstm_layers
        if isinstance(bidirectional, bool):
            bidirectional = [bidirectional] * self.num_lstm_layers
        if isinstance(proj_size, int):
            proj_size = [proj_size] * self.num_lstm_layers

        if isinstance(dropout_p, float):
            dropout_p = [dropout_p] * self.num_inner_layers
        if isinstance(dropout_inplace, bool):
            dropout_inplace = [dropout_inplace] * self.num_inner_layers
        if isinstance(dropout_first, bool):
            dropout_first = [dropout_first] * self.num_inner_layers

        if isinstance(layer_norm, bool):
            layer_norm = [layer_norm] * self.num_inner_layers
        if isinstance(layer_norm_eps, float):
            layer_norm_eps = [layer_norm_eps] * self.num_inner_layers
        if isinstance(layer_norm_element_wise_affine, bool):
            layer_norm_element_wise_affine = [layer_norm_element_wise_affine] * self.num_inner_layers
        if isinstance(layer_norm_bias, bool):
            layer_norm_bias = [layer_norm_bias] * self.num_inner_layers

        assert len(feature_size) == self.num_lstm_layers + 1, 'feature_size must have length num_layers + 1'
        assert len(bias) == self.num_lstm_layers, 'bias must have length num_layers'
        assert len(bidirectional) == self.num_lstm_layers, 'bidirectional must have length num_layers'
        assert len(proj_size) == self.num_lstm_layers, 'proj_size must have length num_layers'

        assert len(dropout_p) == self.num_inner_layers, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == self.num_inner_layers, 'dropout_inplace must have length num_layers - 1'
        assert len(dropout_first) == self.num_inner_layers, 'dropout_first must have length num_layers - 1'

        assert len(layer_norm) == self.num_inner_layers, 'layer_norm must have length num_layers - 1'
        assert len(layer_norm_eps) == self.num_inner_layers, 'layer_norm_eps must have length num_layers - 1'
        assert len(layer_norm_element_wise_affine) == self.num_inner_layers, 'layer_norm_element_wise_affine must have length num_layers - 1'
        assert len(layer_norm_bias) == self.num_inner_layers, 'layer_norm_bias must have length num_layers - 1'

        self.lstm_feature_sizes = feature_size
        self.lstm_bias = bias
        self.lstm_batch_first = batch_first
        self.lstm_bidirectional = bidirectional
        self.lstm_proj_size = proj_size

        self.dropout_p = dropout_p
        self.dropout_inplace = dropout_inplace
        self.dropout_first = dropout_first

        self.layer_norm = layer_norm
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_element_wise_affine = layer_norm_element_wise_affine
        self.layer_norm_bias = layer_norm_bias

        self.lstm_modules = nn.ModuleList()
        self.dropout_modules = nn.ModuleList()
        self.layer_norm_modules = nn.ModuleList()

        for i in range(num_layers):
            self.lstm_modules.append(nn.LSTM(input_size=feature_size[i],
                                        hidden_size=feature_size[i + 1],
                                        num_layers=1,
                                        bias=bias[i],
                                        batch_first=batch_first,
                                        dropout=0.0,
                                        bidirectional=bidirectional[i],
                                        proj_size=proj_size[i]))
            if i >= self.num_inner_layers:
                continue
            if dropout_p[i] > 0.0:
                self.dropout_modules.append(nn.Dropout(p=dropout_p[i],
                                                       inplace=dropout_inplace[i]))
            else:
                self.dropout_modules.append(nn.Identity())
            if layer_norm[i]:
                self.layer_norm_modules.append(nn.LayerNorm(normalized_shape=feature_size[i + 1],
                                                            eps=layer_norm_eps[i],
                                                            elementwise_affine=layer_norm_element_wise_affine[i],
                                                            bias=layer_norm_bias[i]))
            else:
                self.layer_norm_modules.append(nn.Identity())

    def forward(self,
                x: Tensor,
                hx: tuple[Tensor, Tensor] = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:

        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"LSTM: Expected input to be 2D or 3D, got {x.dim()}D instead")
        is_batched = x.dim() == 3
        batch_dim = 0 if self.lstm_batch_first else 1
        if not is_batched:
            x = x.unsqueeze(batch_dim)




        h_n, c_n = None, None
        for i in range(self.num_lstm_layers):
            hx_i = None if hx is None else
            x, (h_i, c_i) = self.lstm_modules[i](x, hx)
            if i >= self.num_inner_layers:
                continue
            if self.dropout_first[i]:
                x = self.dropout_modules[i](x)
                x = self.layer_norm_modules[i](x)
            else:
                x = self.layer_norm_modules[i](x)
                x = self.dropout_modules[i](x)
        return x, (h_n, c_n)

    def get_h_0_c_0(self,
                    batch_size: int,
                    device: torch.device = None,
                    dtype: torch.dtype = None) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        h_0, c_0 = [], []
        for i in range(self.num_lstm_layers):
            h_0_i = torch.zeros(

            )

    @property
    def num_directions(self) -> tuple[int, ...]:
        return tuple(2 if bidirectional else 1 for bidirectional in self.lstm_bidirectional)