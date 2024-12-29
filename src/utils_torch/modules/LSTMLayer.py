import torch
from torch import nn, Tensor


class LSTMLayer(nn.Module):
    _lstm: nn.LSTM

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
        super(LSTMLayer, self).__init__()
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
        self._lstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             proj_size=proj_size,
                             bidirectional=bidirectional,
                             num_layers=1,
                             dropout=0.0,
                             bias=bias,
                             batch_first=batch_first,
                             device=device,
                             dtype=dtype)

    @property
    def input_size(self) -> int:
        return self._lstm.input_size

    @property
    def hidden_size(self) -> int:
        return self._lstm.hidden_size

    @property
    def proj_size(self) -> int:
        return self._lstm.proj_size

    @property
    def output_size(self) -> int:
        return self.hidden_size if self.proj_size == 0 else self.proj_size

    @property
    def bidirectional(self) -> bool:
        return self._lstm.bidirectional

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def batch_first(self) -> bool:
        return self._lstm.batch_first

    @property
    def device(self) -> torch.device:
        return next(self._lstm.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self._lstm.parameters()).dtype

    @property
    def forward_input_weights(self) -> Tensor:
        return self._lstm.weight_ih_l0

    @property
    def forward_hidden_weights(self) -> Tensor:
        return self._lstm.weight_hh_l0

    @property
    def forward_input_bias(self) -> Tensor:
        return self._lstm.bias_ih_l0

    @property
    def forward_hidden_bias(self) -> Tensor:
        return self._lstm.bias_hh_l0

    @property
    def forward_projection_weights(self) -> Tensor | None:
        if self.proj_size == 0:
            return None
        return self._lstm.weight_hr_l0

    @property
    def backward_input_weights(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm.weight_ih_l0_reverse

    @property
    def backward_hidden_weights(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm.weight_hh_l0_reverse

    @property
    def backward_input_bias(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm.bias_ih_l0_reverse

    @property
    def backward_hidden_bias(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        return self._lstm.bias_hh_l0_reverse

    @property
    def backward_projection_weights(self) -> Tensor | None:
        if not self.bidirectional:
            return None
        if self.proj_size == 0:
            return None
        return self._lstm.weight_hr_l0_reverse

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
        self._lstm.weight_ih_l0.data = forward_input_weights
        self._lstm.weight_hh_l0.data = forward_hidden_weights
        self._lstm.bias_ih_l0.data = forward_input_bias
        self._lstm.bias_hh_l0.data = forward_hidden_bias
        if self.proj_size != 0:
            self._lstm.weight_hr_l0.data = forward_projection_weights
        if self.bidirectional:
            self._lstm.weight_ih_l0_reverse.data = backward_input_weights
            self._lstm.weight_hh_l0_reverse.data = backward_hidden_weights
            self._lstm.bias_ih_l0_reverse.data = backward_input_bias
            self._lstm.bias_hh_l0_reverse.data = backward_hidden_bias
            if self.proj_size != 0:
                self._lstm.weight_hr_l0_reverse.data = backward_projection_weights

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

    def check_hx(self, hx: tuple[Tensor, Tensor], batch_size: int):
        h_0, c_0 = hx
        h_0_size, c_0_size = self.get_hx_size(batch_size)
        if h_0.shape != h_0_size:
            raise ValueError(f"Expected hidden state shape {h_0_size}, got {h_0.shape}")
        if c_0.shape != c_0_size:
            raise ValueError(f"Expected cell state shape {c_0_size}, got {c_0.shape}")

    def forward(self,
                x: Tensor,
                hx: tuple[Tensor, Tensor] = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"Expected input tensor to have 2 or 3 dimensions, got {x.dim()}")
        if x.dim() == 2:
            x = x.unsqueeze(self._x_batch_dim)
        if x.size(self._x_feature_dim) != self.input_size:
            raise ValueError(f"Expected input tensor to have feature size {self.input_size}, got {x.size(self._x_feature_dim)}")
        batch_size = x.size(self._x_batch_dim)

        if hx is None:
            hx = self.get_hx(batch_size)
        self.check_hx(hx, batch_size)
        return self._lstm.forward(x, hx)
