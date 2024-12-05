import torch
from torch import Tensor, nn


class LSTMLayerHiddenCell(nn.Module):
    _bidirectional: bool
    _lstm_forward: nn.LSTM
    _lstm_backward: nn.LSTM | None

    _batch_first: bool
    _x_batch_dim: int
    _x_sequence_dim: int
    _x_feature_dim: int
    _hx_direction_dim: int
    _hx_batch_dim: int
    _hx_hidden_dim: int
    _y_batch_dim: int
    _y_sequence_dim: int
    _y_feature_dim: int
    _hy_direction_dim: int
    _hy_batch_dim: int
    _hy_hidden_dim: int

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 proj_size: int = 0,
                 bidirectional: bool = False,
                 bias: bool = True,
                 batch_first: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        super(LSTMLayerHiddenCell, self).__init__()
        self._batch_first = batch_first
        self._x_batch_dim = 0 if batch_first else 1
        self._x_sequence_dim = 1 if batch_first else 0
        self._x_feature_dim = 2
        self._hx_direction_dim = 0
        self._hx_batch_dim = 1
        self._hx_hidden_dim = 2
        self._y_batch_dim = 0 if batch_first else 1
        self._y_sequence_dim = 1 if batch_first else 0
        self._y_feature_dim = 2
        self._hy_direction_dim = 0
        self._hy_batch_dim = 1
        self._hy_hidden_dim = 2
        self._bidirectional = bidirectional
        self._lstm_forward = torch.nn.LSTM(input_size=input_size,
                                           hidden_size=hidden_size,
                                           proj_size=proj_size,
                                           bidirectional=False,
                                           num_layers=1,
                                           dropout=0.0,
                                           bias=bias,
                                           batch_first=batch_first,
                                           device=device,
                                           dtype=dtype)
        self._lstm_backward = torch.nn.LSTM(input_size=input_size,
                                            hidden_size=hidden_size,
                                            proj_size=proj_size,
                                            bidirectional=False,
                                            num_layers=1,
                                            dropout=0.0,
                                            bias=bias,
                                            batch_first=batch_first,
                                            device=device,
                                            dtype=dtype) if bidirectional else None

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def input_size(self) -> int:
        return self._lstm_forward.input_size

    @property
    def hidden_size(self) -> int:
        return self._lstm_forward.hidden_size

    @property
    def proj_size(self) -> int:
        return self._lstm_forward.proj_size

    @property
    def output_size(self) -> int:
        output_size = self.hidden_size if self.proj_size == 0 else self.proj_size
        output_size *= self.num_directions
        return output_size

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    @property
    def bias(self) -> bool:
        return self._lstm_forward.bias

    @property
    def batch_first(self) -> bool:
        return self._batch_first


    @property
    def device(self) -> torch.device:
        return next(self._lstm_forward.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._lstm_forward.parameters()).dtype

    @property
    def forward_input_weights(self) -> Tensor:
        return self._lstm_forward.weight_ih_l0

    @property
    def forward_hidden_weights(self) -> Tensor:
        return self._lstm_forward.weight_hh_l0

    @property
    def forward_input_bias(self) -> Tensor:
        return self._lstm_forward.bias_ih_l0

    @property
    def forward_hidden_bias(self) -> Tensor:
        return self._lstm_forward.bias_hh_l0

    @property
    def forward_projection_weights(self) -> Tensor | None:
        if self.proj_size == 0:
            return None
        return self._lstm_forward.weight_hr_l0

    @property
    def backward_input_weights(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm_backward.weight_ih_l0

    @property
    def backward_hidden_weights(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm_backward.weight_hh_l0

    @property
    def backward_input_bias(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm_backward.bias_ih_l0

    @property
    def backward_hidden_bias(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm_backward.bias_hh_l0

    @property
    def backward_projection_weights(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        if self.proj_size == 0:
            return None
        return self._lstm_backward.weight_hr_l0

    def get_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        return (
            self.forward_input_weights,
            self.forward_hidden_weights,
            self.forward_input_bias,
            self.forward_hidden_bias,
            self.forward_projection_weights,
            self.backward_input_weights,
            self.backward_hidden_weights,
            self.backward_input_bias,
            self.backward_hidden_bias,
            self.backward_projection_weights
        )

    def set_parameters(self, parameters: tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None]):
        forward_input_weights, forward_hidden_weights, forward_input_bias, forward_hidden_bias, forward_projection_weights, backward_input_weights, backward_hidden_weights, backward_input_bias, backward_hidden_bias, backward_projection_weights = parameters
        self._lstm_forward.weight_ih_l0.data = forward_input_weights
        self._lstm_forward.weight_hh_l0.data = forward_hidden_weights
        self._lstm_forward.bias_ih_l0.data = forward_input_bias
        self._lstm_forward.bias_hh_l0.data = forward_hidden_bias
        if self.proj_size != 0:
            self._lstm_forward.weight_hr_l0.data = forward_projection_weights
        if self.bidirectional:
            self._lstm_backward.weight_ih_l0.data = backward_input_weights
            self._lstm_backward.weight_hh_l0.data = backward_hidden_weights
            self._lstm_backward.bias_ih_l0.data = backward_input_bias
            self._lstm_backward.bias_hh_l0.data = backward_hidden_bias
            if self.proj_size != 0:
                self._lstm_backward.weight_hr_l0.data = backward_projection_weights

    def get_hx_size(self, batch_size: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        h_0_size = self.num_directions, batch_size, self.hidden_size if self.proj_size == 0 else self.proj_size
        c_0_size = self.num_directions, batch_size, self.hidden_size
        return h_0_size, c_0_size

    def get_hx(self, batch_size: int) -> tuple[Tensor, Tensor]:
        device = self.device
        dtype = self.dtype
        h_0_size, c_0_size = self.get_hx_size(batch_size)
        h_0 = torch.zeros(h_0_size, device=device, dtype=dtype)
        c_0 = torch.zeros(c_0_size, device=device, dtype=dtype)
        return h_0, c_0

    def check_hx(self, hx: tuple[Tensor, Tensor], batch_size: int) -> None:
        h_0, c_0 = hx
        h_0_size, c_0_size = self.get_hx_size(batch_size)
        if h_0.size() != h_0_size:
            raise ValueError(f"Expected hidden state size {h_0_size}, got {h_0.size()}")
        if c_0.size() != c_0_size:
            raise ValueError(f"Expected cell state size {c_0_size}, got {c_0.size()}")

    def forward(self,
                x: Tensor,
                hx: tuple[Tensor, Tensor] = None) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"Expected input tensor to have 2 or 3 dimensions, got {x.dim()}")
        if x.dim() == 2:
            x = x.unsqueeze(self._x_batch_dim)
        if x.size(self._x_feature_dim) != self.input_size:
            raise ValueError(f"Expected input feature size {self.input_size}, got {x.size(self._x_feature_dim)}")
        batch_size = x.size(self._x_batch_dim)
        seq_size = x.size(self._x_sequence_dim)

        if hx is None:
            hx = self.get_hx(batch_size)
        self.check_hx(hx, batch_size)
        h_0, c_0 = hx

        device = next(self._lstm_forward.parameters()).device
        forward_idx = torch.tensor(0).to(device=device)
        backward_idx = torch.tensor(1).to(device=device)
        seq_idx = torch.tensor(range(seq_size)).to(device=device)

        hy_forward, cy_forward = [], []
        h_i_forward = h_0.index_select(self._hx_direction_dim, forward_idx)
        c_i_forward = c_0.index_select(self._hx_direction_dim, forward_idx)
        for i in range(seq_size):
            x_i_forward = x.index_select(self._x_sequence_dim, seq_idx[i])
            _, (h_i_forward, c_i_forward) = self._lstm_forward(x_i_forward, (h_i_forward, c_i_forward))
            hy_forward.append(h_i_forward)
            cy_forward.append(c_i_forward)

        if not self.bidirectional:
            y_hidden = torch.stack(hy_forward, dim=self._y_sequence_dim)
            y_cell = torch.stack(cy_forward, dim=self._y_sequence_dim)
            last_hidden = h_i_forward
            last_cell = c_i_forward
            return (y_hidden, y_cell), (last_hidden, last_cell)

        hy_backward, cy_backward = [], []
        h_i_backward = h_0.index_select(self._hx_direction_dim, backward_idx)
        c_i_backward = c_0.index_select(self._hx_direction_dim, backward_idx)
        for i in reversed(range(seq_size)):
            x_i_backward = x.index_select(self._x_sequence_dim, seq_idx[i])
            _, (h_i_backward, c_i_backward) = self._lstm_backward(x_i_backward, (h_i_backward, c_i_backward))
            hy_backward.append(h_i_backward)
            cy_backward.append(c_i_backward)

        y_hidden_forward = torch.stack(hy_forward, dim=self._y_sequence_dim)
        y_hidden_backward = torch.stack(hy_backward, dim=self._y_sequence_dim)
        y_hidden = torch.cat((y_hidden_forward, y_hidden_backward), dim=self._y_feature_dim)
        y_cell_forward = torch.stack(cy_forward, dim=self._y_sequence_dim)
        y_cell_backward = torch.stack(cy_backward, dim=self._y_sequence_dim)
        y_cell = torch.cat((y_cell_forward, y_cell_backward), dim=self._y_feature_dim)
        last_hidden = torch.cat((h_i_forward, h_i_backward), dim=self._hy_direction_dim)
        last_cell = torch.cat((c_i_forward, c_i_backward), dim=self._hy_direction_dim)
        return (y_hidden, y_cell), (last_hidden, last_cell)
