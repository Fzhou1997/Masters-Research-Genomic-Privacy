from typing import Sequence, Union

import torch
from torch import nn, Size, Tensor


_shape_t = Union[int, list[int], Size]

class MultiLayerLSTM(nn.Module):
    """
    A multi-layer LSTM module with optional projection, dropout, and layer normalization.

    Attributes:
        num_lstm_layers (int): Number of LSTM layers.
        num_inner_layers (int): Number of inner layers (num_lstm_layers - 1).
        lstm_input_size (Sequence[int]): Input size for each LSTM layer.
        lstm_hidden_size (Sequence[int]): Hidden size for each LSTM layer.
        lstm_proj_size (Sequence[int]): Projection size for each LSTM layer.
        lstm_output_size (Sequence[int]): Output size for each LSTM layer.
        lstm_bidirectional (Sequence[bool]): Whether each LSTM layer is bidirectional.
        lstm_num_directions (Sequence[int]): Number of directions for each LSTM layer.
        lstm_bias (Sequence[bool]): Whether each LSTM layer uses bias.
        lstm_batch_first (bool): Whether the input and output tensors are provided as (batch, seq, feature).
        dropout_p (Sequence[float]): Dropout probability for each inner layer.
        dropout_inplace (Sequence[bool]): Whether to perform dropout in-place for each inner layer.
        dropout_first (Sequence[bool]): Whether to apply dropout before layer normalization for each inner layer.
        layer_norm (Sequence[bool]): Whether to apply layer normalization for each inner layer.
        layer_norm_eps (Sequence[float]): Epsilon value for layer normalization.
        layer_norm_element_wise_affine (Sequence[bool]): Whether to apply element-wise affine transformation in layer normalization.
        layer_norm_bias (Sequence[bool]): Whether to use bias in layer normalization.
        lstm_modules (nn.ModuleList): List of LSTM modules.
        dropout_modules (nn.ModuleList): List of dropout modules.
        layer_norm_modules (nn.ModuleList): List of layer normalization modules.
    """

    num_lstm_layers: int
    num_inner_layers: int

    lstm_input_size: Sequence[int]
    lstm_hidden_size: Sequence[int]
    lstm_proj_size: Sequence[int]
    lstm_output_size: Sequence[int]
    lstm_bidirectional: Sequence[bool]
    lstm_num_directions: Sequence[int]
    lstm_bias: Sequence[bool]
    lstm_batch_first: bool

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
                 input_size: int,
                 hidden_size: int | Sequence[int],
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
        """
        Initializes the MultiLayerLSTM module.

        Args:
            num_layers (int): Number of LSTM layers.
            input_size (int): Size of the input features.
            hidden_size (int or Sequence[int]): Size of the hidden state for each LSTM layer.
            proj_size (int or Sequence[int], optional): Size of the projection for each LSTM layer. Defaults to 0.
            bidirectional (bool or Sequence[bool], optional): Whether each LSTM layer is bidirectional. Defaults to False.
            bias (bool or Sequence[bool], optional): Whether each LSTM layer uses bias. Defaults to True.
            batch_first (bool, optional): Whether the input and output tensors are provided as (batch, seq, feature). Defaults to True.
            dropout_p (float or Sequence[float], optional): Dropout probability for each inner layer. Defaults to 0.5.
            dropout_inplace (bool or Sequence[bool], optional): Whether to perform dropout in-place for each inner layer. Defaults to True.
            dropout_first (bool or Sequence[bool], optional): Whether to apply dropout before layer normalization for each inner layer. Defaults to True.
            layer_norm (bool or Sequence[bool], optional): Whether to apply layer normalization for each inner layer. Defaults to True.
            layer_norm_eps (float or Sequence[float], optional): Epsilon value for layer normalization. Defaults to 1e-5.
            layer_norm_element_wise_affine (bool or Sequence[bool], optional): Whether to apply element-wise affine transformation in layer normalization. Defaults to True.
            layer_norm_bias (bool or Sequence[bool], optional): Whether to use bias in layer normalization. Defaults to True.
        """
        super(MultiLayerLSTM, self).__init__()

        self.num_lstm_layers = num_layers
        self.num_inner_layers = num_layers - 1

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size] * self.num_lstm_layers
        if isinstance(proj_size, int):
            proj_size = [proj_size] * self.num_lstm_layers
        if isinstance(bidirectional, bool):
            bidirectional = [bidirectional] * self.num_lstm_layers
        if isinstance(bias, bool):
            bias = [bias] * self.num_lstm_layers

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

        assert len(hidden_size) == self.num_lstm_layers, 'hidden_size must have length num_layers'
        assert len(proj_size) == self.num_lstm_layers, 'proj_size must have length num_layers'
        assert len(bidirectional) == self.num_lstm_layers, 'bidirectional must have length num_layers'
        assert len(bias) == self.num_lstm_layers, 'bias must have length num_layers'

        assert len(dropout_p) == self.num_inner_layers, 'dropout_p must have length num_layers - 1'
        assert len(dropout_inplace) == self.num_inner_layers, 'dropout_inplace must have length num_layers - 1'
        assert len(dropout_first) == self.num_inner_layers, 'dropout_first must have length num_layers - 1'

        assert len(layer_norm) == self.num_inner_layers, 'layer_norm must have length num_layers - 1'
        assert len(layer_norm_eps) == self.num_inner_layers, 'layer_norm_eps must have length num_layers - 1'
        assert len(layer_norm_element_wise_affine) == self.num_inner_layers, 'layer_norm_element_wise_affine must have length num_layers - 1'
        assert len(layer_norm_bias) == self.num_inner_layers, 'layer_norm_bias must have length num_layers - 1'

        output_size = []
        for i in range(self.num_lstm_layers):
            output_size_i = proj_size[i] if proj_size[i] > 0 else hidden_size[i]
            output_size_i *= 2 if bidirectional[i] else 1
            output_size.append(output_size_i)
        self.lstm_output_size = output_size

        input_sizes = [input_size] + output_size[:-1]
        self.lstm_input_size = input_sizes

        self.lstm_hidden_size = hidden_size
        self.lstm_proj_size = proj_size
        self.lstm_bidirectional = bidirectional
        self.lstm_num_directions = [2 if bidirectional_i else 1 for bidirectional_i in bidirectional]
        self.lstm_bias = bias
        self.lstm_batch_first = batch_first

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
            self.lstm_modules.append(nn.LSTM(input_size=input_sizes[i],
                                             hidden_size=hidden_size[i],
                                             proj_size=proj_size[i],
                                             bidirectional=bidirectional[i],
                                             bias=bias[i],
                                             batch_first=batch_first,
                                             num_layers=1,
                                             dropout=0.0))
            if i >= self.num_inner_layers:
                continue
            if dropout_p[i] > 0.0:
                self.dropout_modules.append(nn.Dropout(p=dropout_p[i],
                                                       inplace=dropout_inplace[i]))
            else:
                self.dropout_modules.append(nn.Identity())
            output_size_i = proj_size[i] if proj_size[i] > 0 else hidden_size[i]
            output_size_i *= 2 if bidirectional[i] else 1
            if layer_norm[i]:
                self.layer_norm_modules.append(nn.LayerNorm(normalized_shape=output_size_i,
                                                            eps=layer_norm_eps[i],
                                                            elementwise_affine=layer_norm_element_wise_affine[i],
                                                            bias=layer_norm_bias[i]))
            else:
                self.layer_norm_modules.append(nn.Identity())

    def forward(self,
                x: Tensor,
                hx: tuple[tuple[Tensor, ...], tuple[Tensor, ...]] = None) -> tuple[Tensor, tuple[tuple[Tensor, ...], tuple[Tensor, ...]]]:
        """
        Forward pass for the MultiLayerLSTM module.

        Args:
            x (Tensor): Input tensor of shape (batch, seq, feature) if batch_first, otherwise (seq, batch, feature).
            hx (tuple[tuple[Tensor, ...], tuple[Tensor, ...]], optional): Initial hidden and cell states. Defaults to None.

        Returns:
            tuple[Tensor, tuple[tuple[Tensor, ...], tuple[Tensor, ...]]]: Output tensor and the final hidden and cell states.
        """
        self.check_x(x)
        is_batched = x.dim() == 3
        batch_dim = 0 if self.lstm_batch_first else 1
        if not is_batched:
            x = x.unsqueeze(batch_dim)
        batch_size = x.size(batch_dim)
        if hx is None:
            hx = self.get_hx(batch_size, x.device, x.dtype)
        self.check_hx(hx)

        hy, cy = [], []
        for i in range(self.num_lstm_layers):
            hx_i = (hx[0][i].contiguous(), hx[1][i].contiguous())
            x, (hy_i, cy_i) = self.lstm_modules[i](x, hx_i)
            hy.append(hy_i)
            cy.append(cy_i)
            if i >= self.num_inner_layers:
                continue
            if self.dropout_first[i]:
                x = self.dropout_modules[i](x)
                x = self.layer_norm_modules[i](x)
            else:
                x = self.layer_norm_modules[i](x)
                x = self.dropout_modules[i](x)
        y = x
        return y, (tuple(hy), tuple(cy))

    def get_hx_size(self,
                    batch_size: int) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
        """
        Get the size of the hidden and cell states.

        Args:
            batch_size (int): Batch size.

        Returns:
            tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]: Sizes of the hidden and cell states.
        """
        h_0_size = []
        c_0_size = []
        for i in range(self.num_lstm_layers):
            h_0_size_i = (self.lstm_num_directions[i],
                          batch_size,
                          self.lstm_output_size[i])
            c_0_size_i = (self.lstm_num_directions[i],
                          batch_size,
                          self.lstm_hidden_size[i])
            h_0_size.append(h_0_size_i)
            c_0_size.append(c_0_size_i)
        return tuple(h_0_size), tuple(c_0_size)

    def get_hx(self,
               batch_size: int,
               device: torch.device = None,
               dtype: torch.dtype = None) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        """
        Get the initial hidden and cell states.

        Args:
            batch_size (int): Batch size.
            device (torch.device, optional): Device for the tensors. Defaults to None.
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to None.

        Returns:
            tuple[tuple[Tensor, ...], tuple[Tensor, ...]]: Initial hidden and cell states.
        """
        hx_size = self.get_hx_size(batch_size)
        h_0 = [torch.zeros(h_0_size_i, device=device, dtype=dtype) for h_0_size_i in hx_size[0]]
        c_0 = [torch.zeros(c_0_size_i, device=device, dtype=dtype) for c_0_size_i in hx_size[1]]
        return tuple(h_0), tuple(c_0)

    def check_hx(self, hx: tuple[tuple[Tensor, ...], tuple[Tensor, ...]]) -> None:
        """
        Check the validity of the hidden and cell states.

        Args:
            hx (tuple[tuple[Tensor, ...], tuple[Tensor, ...]]): Hidden and cell states.

        Raises:
            ValueError: If the hidden and cell states are not valid.
        """
        batch_size = hx[0][0].size(1)
        hx_size = self.get_hx_size(batch_size)
        if len(hx) != len(hx_size):
            raise ValueError(f"MultiLayerLSTM: Expected hx to be a tuple of length {len(hx_size)}, got length {len(hx)} instead")
        if len(hx[0]) != len(hx_size[0]):
            raise ValueError(f"MultiLayerLSTM: Expected h_0 to be a tuple of length {len(hx_size[0])}, got length {len(hx[0])} instead")
        if len(hx[1]) != len(hx_size[1]):
            raise ValueError(f"MultiLayerLSTM: Expected c_0 to be a tuple of length {len(hx_size[1])}, got length {len(hx[1])} instead")
        for i in range(self.num_lstm_layers):
            h_0_i = hx[0][i]
            c_0_i = hx[1][i]
            h_0_i_size = hx_size[0][i]
            c_0_i_size = hx_size[1][i]
            if h_0_i.dim() != len(h_0_i_size):
                raise ValueError(f"MultiLayerLSTM: Expected h_0 to be {len(h_0_i_size)}D, got {h_0_i.dim()}D instead")
            if c_0_i.dim() != len(c_0_i_size):
                raise ValueError(f"MultiLayerLSTM: Expected c_0 to be {len(c_0_i_size)}D, got {c_0_i.dim()}D instead")
            if h_0_i.size() != h_0_i_size:
                raise ValueError(f"MultiLayerLSTM: Expected h_0 to have size {h_0_i_size}, got size {h_0_i.size()} instead")
            if c_0_i.size() != c_0_i_size:
                raise ValueError(f"MultiLayerLSTM: Expected c_0 to have size {c_0_i_size}, got size {c_0_i.size()} instead")
            if h_0_i.size(1) != batch_size:
                raise ValueError(f"MultiLayerLSTM: Expected h_0 to have size {batch_size} in the batch dimension, got size {h_0_i.size(1)} instead")
            if c_0_i.size(1) != batch_size:
                raise ValueError(f"MultiLayerLSTM: Expected c_0 to have size {batch_size} in the batch dimension, got size {c_0_i.size(1)} instead")

    def check_x(self,
                x: Tensor) -> None:
        """
        Check the validity of the input tensor.

        Args:
            x (Tensor): Input tensor.

        Raises:
            ValueError: If the input tensor is not valid.
        """
        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"MultiLayerLSTM: Expected input to be 2D or 3D, got {x.dim()}D instead")
        if x.size(-1) != self.lstm_input_size[0]:
            raise ValueError(f"MultiLayerLSTM: Expected input to have size {self.lstm_input_size[0]} in the feature dimension, got size {x.size(-1)} instead")
