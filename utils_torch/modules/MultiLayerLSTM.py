from typing import Sequence, Self, Optional

import torch
from torch import nn, Tensor
from torch._prims_common import DeviceLikeType

hx_type = tuple[tuple[Tensor, ...], tuple[Tensor, ...]]

y_type = tuple[Tensor, tuple[tuple[Tensor, ...], tuple[Tensor, ...]]]

class MultiLayerLSTM(nn.Module):
    """
    A multi-layer LSTM module with optional projection, dropout, and layer normalization.

    Attributes:
        _lstm_num_layers (int): Number of LSTM layers.
        _inner_num_layers (int): Number of inner layers (num_lstm_layers - 1).

        _lstm_input_size (tuple[int]): Input size for each LSTM layer.
        _lstm_hidden_size (tuple[int]): Hidden size for each LSTM layer.
        _lstm_proj_size (tuple[int]): Projection size for each LSTM layer.
        _lstm_output_size (tuple[int]): Output size for each LSTM layer.
        _lstm_bidirectional (tuple[bool]): Whether each LSTM layer is bidirectional.
        _lstm_num_directions (tuple[int]): Number of directions for each LSTM layer.
        _lstm_bias (tuple[bool]): Whether each LSTM layer uses bias.
        _lstm_batch_first (bool): Whether the input and output tensors are provided as (batch, seq, feature).

        _dropout_p (tuple[float]): Dropout probability for each inner layer.
        _dropout_inplace (tuple[bool]): Whether to perform dropout in-place for each inner layer.
        _dropout_first (tuple[bool]): Whether to apply dropout before layer normalization for each inner layer.

        _layer_norm (tuple[bool]): Whether to apply layer normalization for each inner layer.
        _layer_norm_eps (tuple[float]): Epsilon value for layer normalization.
        _layer_norm_element_wise_affine (tuple[bool]): Whether to apply element-wise affine transformation in layer normalization.
        _layer_norm_bias (tuple[bool]): Whether to use bias in layer normalization.

        _lstm_modules (nn.ModuleList): List of LSTM modules.
        _dropout_modules (nn.ModuleList): List of dropout modules.
        _layer_norm_modules (nn.ModuleList): List of layer normalization modules.
    """

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

    _device: torch.device
    _dtype: torch.dtype

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
            dropout_first (bool or Sequence[bool], optional): Whether to apply dropout before layer normalization for each inner layer. Defaults to False.

            layer_norm (bool or Sequence[bool], optional): Whether to apply layer normalization for each inner layer. Defaults to True.
            layer_norm_eps (float or Sequence[float], optional): Epsilon value for layer normalization. Defaults to 1e-5.
            layer_norm_element_wise_affine (bool or Sequence[bool], optional): Whether to apply element-wise affine transformation in layer normalization. Defaults to True.
            layer_norm_bias (bool or Sequence[bool], optional): Whether to use bias in layer normalization. Defaults to True.

            device (torch.device, optional): Device for the tensors. Defaults to None.
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to None.
        """
        super(MultiLayerLSTM, self).__init__()

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
            layer_norm_element_wise_affine = [layer_norm_element_wise_affine] * self._inner_num_layers
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
        assert len(layer_norm_element_wise_affine) == self._inner_num_layers, 'layer_norm_element_wise_affine must have length num_layers - 1'
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
        self._lstm_num_directions = tuple([2 if bidirectional_i else 1 for bidirectional_i in bidirectional])
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

        self._device = device
        self._dtype = dtype

        for i in range(num_layers):
            self._lstm_modules.append(nn.LSTM(input_size=input_sizes[i],
                                              hidden_size=hidden_size[i],
                                              proj_size=proj_size[i],
                                              bidirectional=bidirectional[i],
                                              bias=bias[i],
                                              batch_first=batch_first,
                                              num_layers=1,
                                              dropout=0.0,
                                              device=device,
                                              dtype=dtype))
            if i >= self._inner_num_layers:
                continue
            if dropout_p[i] > 0.0:
                self._dropout_modules.append(nn.Dropout(p=dropout_p[i],
                                                        inplace=dropout_inplace[i]))
            else:
                self._dropout_modules.append(nn.Identity())
            output_size_i = proj_size[i] if proj_size[i] > 0 else hidden_size[i]
            output_size_i *= 2 if bidirectional[i] else 1
            if layer_norm[i]:
                self._layer_norm_modules.append(nn.LayerNorm(normalized_shape=output_size_i,
                                                             eps=layer_norm_eps[i],
                                                             elementwise_affine=layer_norm_element_wise_affine[i],
                                                             bias=layer_norm_bias[i],
                                                             device=device,
                                                             dtype=dtype))
            else:
                self._layer_norm_modules.append(nn.Identity())

    def to(self,
           device: Optional[DeviceLikeType] = ...,
           dtype: Optional[torch.dtype] = ...,
           non_blocking: bool = ...,) -> Self:
        """
        Moves and/or casts the parameters and buffers.

        Args:
            device (torch.device, optional): The desired device of the parameters and buffers in this module. Defaults to None.
            dtype (torch.dtype, optional): The desired floating point type of the parameters and buffers in this module. Defaults to None.
            non_blocking (bool, optional): Whether to convert the tensors asynchronously. Defaults to False.

        Returns:
            Self: The module itself.
        """
        if device is not ...:
            self._device = device
        if dtype is not ...:
            self._dtype = dtype
        return super().to(device, dtype, non_blocking)

    def forward(self,
                x: Tensor,
                hx: hx_type = None) -> y_type:
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
        batch_dim = 0 if self._lstm_batch_first else 1
        if not is_batched:
            x = x.unsqueeze(batch_dim)
        batch_size = x.size(batch_dim)
        if hx is None:
            hx = self.get_hx(batch_size, self._device, self._dtype)
        self.check_hx(hx)

        hy, cy = [], []
        for i in range(self._lstm_num_layers):
            hx_i = (hx[0][i].contiguous(), hx[1][i].contiguous())
            x, (hy_i, cy_i) = self._lstm_modules[i](x, hx_i)
            hy.append(hy_i)
            cy.append(cy_i)
            if i >= self._inner_num_layers:
                continue
            if self._dropout_first[i]:
                x = self._dropout_modules[i](x)
                x = self._layer_norm_modules[i](x)
            else:
                x = self._layer_norm_modules[i](x)
                x = self._dropout_modules[i](x)
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
        """
        Get the initial hidden and cell states.

        Args:
            batch_size (int): Batch size.
            device (torch.device, optional): Device for the tensors. Defaults to None.
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to None.

        Returns:
            tuple[tuple[Tensor, ...], tuple[Tensor, ...]]: Initial hidden and cell states.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        hx_size = self.get_hx_size(batch_size)
        h_0 = [torch.zeros(h_0_size_i, device=device, dtype=dtype) for h_0_size_i in hx_size[0]]
        c_0 = [torch.zeros(c_0_size_i, device=device, dtype=dtype) for c_0_size_i in hx_size[1]]
        return tuple(h_0), tuple(c_0)

    def check_hx(self, hx: hx_type) -> None:
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
        for i in range(self._lstm_num_layers):
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
        if x.size(-1) != self._lstm_input_size[0]:
            raise ValueError(f"MultiLayerLSTM: Expected input to have size {self._lstm_input_size[0]} in the feature dimension, got size {x.size(-1)} instead")

    @property
    def lstm_num_layers(self) -> int:
        """
        Returns the number of LSTM layers.

        Returns:
            int: Number of LSTM layers.
        """
        return self._lstm_num_layers

    @property
    def lstm_input_size_in(self) -> int:
        """
        Returns the input size for the first LSTM layer.

        Returns:
            int: Input size for the first LSTM layer.
        """
        return self._lstm_input_size[0]

    @property
    def lstm_output_size_out(self) -> int:
        """
        Returns the output size for the last LSTM layer.

        Returns:
            int: Output size for the last LSTM layer.
        """
        return self._lstm_output_size[-1]

    @property
    def lstm_input_size(self) -> tuple[int, ...]:
        """
        Returns the input sizes for all LSTM layers.

        Returns:
            tuple[int, ...]: Input sizes for all LSTM layers.
        """
        return self._lstm_input_size

    @property
    def lstm_hidden_size(self) -> tuple[int, ...]:
        """
        Returns the hidden sizes for all LSTM layers.

        Returns:
            tuple[int, ...]: Hidden sizes for all LSTM layers.
        """
        return self._lstm_hidden_size

    @property
    def lstm_proj_size(self) -> tuple[int, ...]:
        """
        Returns the projection sizes for all LSTM layers.

        Returns:
            tuple[int, ...]: Projection sizes for all LSTM layers.
        """
        return self._lstm_proj_size

    @property
    def lstm_output_size(self) -> tuple[int, ...]:
        """
        Returns the output sizes for all LSTM layers.

        Returns:
            tuple[int, ...]: Output sizes for all LSTM layers.
        """
        return self._lstm_output_size

    @property
    def lstm_bidirectional(self) -> tuple[bool, ...]:
        """
        Returns whether each LSTM layer is bidirectional.

        Returns:
            tuple[bool, ...]: Whether each LSTM layer is bidirectional.
        """
        return self._lstm_bidirectional

    @property
    def lstm_num_directions(self) -> tuple[int, ...]:
        """
        Returns the number of directions for each LSTM layer.

        Returns:
            tuple[int, ...]: Number of directions for each LSTM layer.
        """
        return self._lstm_num_directions

    @property
    def lstm_bias(self) -> tuple[bool, ...]:
        """
        Returns whether each LSTM layer uses bias.

        Returns:
            tuple[bool, ...]: Whether each LSTM layer uses bias.
        """
        return self._lstm_bias

    @property
    def lstm_batch_first(self) -> bool:
        """
        Returns whether the input and output tensors are provided as (batch, seq, feature).

        Returns:
            bool: Whether the input and output tensors are provided as (batch, seq, feature).
        """
        return self._lstm_batch_first

    @property
    def lstm_dropout_p(self) -> tuple[float, ...]:
        """
        Returns the dropout probabilities for each inner layer.

        Returns:
            tuple[float, ...]: Dropout probabilities for each inner layer.
        """
        return self._dropout_p

    @property
    def lstm_dropout_inplace(self) -> tuple[bool, ...]:
        """
        Returns whether to perform dropout in-place for each inner layer.

        Returns:
            tuple[bool, ...]: Whether to perform dropout in-place for each inner layer.
        """
        return self._dropout_inplace

    @property
    def lstm_dropout_first(self) -> tuple[bool, ...]:
        """
        Returns whether to apply dropout before layer normalization for each inner layer.

        Returns:
            tuple[bool, ...]: Whether to apply dropout before layer normalization for each inner layer.
        """
        return self._dropout_first

    @property
    def lstm_layer_norm(self) -> tuple[bool, ...]:
        """
        Returns whether to apply layer normalization for each inner layer.

        Returns:
            tuple[bool, ...]: Whether to apply layer normalization for each inner layer.
        """
        return self._layer_norm

    @property
    def lstm_layer_norm_eps(self) -> tuple[float, ...]:
        """
        Returns the epsilon value for layer normalization.

        Returns:
            tuple[float, ...]: Epsilon value for layer normalization.
        """
        return self._layer_norm_eps

    @property
    def lstm_layer_norm_element_wise_affine(self) -> tuple[bool, ...]:
        """
        Returns whether to apply element-wise affine transformation in layer normalization.

        Returns:
            tuple[bool, ...]: Whether to apply element-wise affine transformation in layer normalization.
        """
        return self._layer_norm_element_wise_affine

    @property
    def lstm_layer_norm_bias(self) -> tuple[bool, ...]:
        """
        Returns whether to use bias in layer normalization.

        Returns:
            tuple[bool, ...]: Whether to use bias in layer normalization.
        """
        return self._layer_norm_bias
