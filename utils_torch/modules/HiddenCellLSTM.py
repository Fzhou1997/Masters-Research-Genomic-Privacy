import torch
from torch import Tensor, nn


class HiddenCellLSTM(nn.Module):

    lstm: torch.nn.LSTM
    hidden: torch.Tensor
    cell: torch.Tensor

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 proj_size: int = 0,
                 bias: bool = True,
                 batch_first: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        super(HiddenCellLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  proj_size=proj_size,
                                  num_layers=1,
                                  dropout=0.0,
                                  bias=bias,
                                  batch_first=batch_first,
                                  device=device,
                                  dtype=dtype)

    @property
    def input_size(self) -> int:
        return self.lstm.input_size

    @property
    def hidden_size(self) -> int:
        return self.lstm.hidden_size

    @property
    def proj_size(self) -> int:
        return self.lstm.proj_size

    @property
    def output_size(self) -> int:
        return self.hidden_size if self.proj_size == 0 else self.proj_size

    @property
    def bidirectional(self) -> bool:
        return self.lstm.bidirectional

    @property
    def num_directions(self) -> int:
        return 2 if self.bidirectional else 1

    @property
    def batch_first(self) -> bool:
        return self.lstm.batch_first

    @property
    def batch_dim(self) -> int:
        return 0 if self.batch_first else 1

    @property
    def sequence_dim(self) -> int:
        return 1 if self.batch_first else 0

    @property
    def feature_dim(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return self.lstm.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lstm.dtype

    def get_hx_size(self, batch_size: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        h_0_size = self.num_directions, batch_size, self.hidden_size if self.proj_size == 0 else self.proj_size
        c_0_size = self.num_directions, batch_size, self.hidden_size
        return h_0_size, c_0_size

    def get_hx(self, batch_size: int) -> tuple[Tensor, Tensor]:
        h_0_size, c_0_size = self.get_hx_size(batch_size)
        h_0 = torch.zeros(h_0_size, device=self.lstm.device, dtype=self.lstm.dtype)
        c_0 = torch.zeros(c_0_size, device=self.lstm.device, dtype=self.lstm.dtype)
        return h_0, c_0

    def check_hx(self, hx: tuple[Tensor, Tensor], batch_size: int) -> None:
        h_0, c_0 = hx
        h_0_size, c_0_size = self.get_hx_size(batch_size)
        if h_0.size() != h_0_size:
            raise ValueError(f"Expected hidden state size {h_0_size}, got {h_0.size()}")
        if c_0.size() != c_0_size:
            raise ValueError(f"Expected cell state size {c_0_size}, got {c_0.size()}")

    def check_x(self, x: Tensor) -> None:
        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"Expected input tensor of size 3 or 2, got {x.dim()}")
        if x.size(self.feature_dim) != self.input_size:
            raise ValueError(f"Expected input feature size {self.input_size}, got {x.size(self.feature_dim)}")

    def forward(self, x: Tensor, hx: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        self.check_x(x)
        if x.dim() == 2:
            x = x.unsqueeze(self.batch_dim)
        batch_size = x.size(self.batch_dim)
        if hx is None:
            hx = self.get_hx(batch_size)
        self.check_hx(hx, batch_size)
        hy, cy = [], []
        h_i, c_i = hx
        for i in range(x.size(self.sequence_dim)):
            x_i = x.select(self.sequence_dim, i)
            _, (h_i, c_i) = self.lstm(x_i, (h_i, c_i))
            hy.append(h_i)
            cy.append(c_i)
        hy, cy = torch.stack(hy, dim=0).contiguous(), torch.stack(cy, dim=0).contiguous()
        return hy, cy, (hy[-1], cy[-1])

