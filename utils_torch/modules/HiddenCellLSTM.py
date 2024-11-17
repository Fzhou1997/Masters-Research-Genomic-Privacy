import torch
from torch import Tensor


class HiddenCellLSTM:

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
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional,
                                  proj_size=proj_size,
                                  num_layers=1,
                                  bias=bias,
                                  batch_first=batch_first,
                                  device=device,
                                  dtype=dtype)

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

    def forward(self, x: Tensor, hx: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        hy, cy = [], []
        for i in range(x.size(self.sequence_dim)):
            x_i = x.select(self.sequence_dim, i)
            _, (h_i, c_i) = self.lstm
            pass
        hy, cy = torch.stack(hy, dim=0).contiguous(), torch.stack(cy, dim=0).contiguous()
        return hy, cy, (hy[-1], cy[-1])

