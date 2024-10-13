import os
from typing import Self

import torch
import torch.nn as nn
from torch import Tensor


class LSTMAttacker(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0.0,
                 output_size: int = 1):
        super(LSTMAttacker, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    @property
    def input_size(self) -> int:
        return self.lstm.input_size

    @property
    def hidden_size(self) -> int:
        return self.lstm.hidden_size

    @property
    def num_layers(self) -> int:
        return self.lstm.num_layers

    @property
    def num_directions(self) -> int:
        return 2 if self.lstm.bidirectional else 1

    @property
    def dropout(self) -> float:
        return self.lstm.dropout

    @property
    def is_bidirectional(self) -> bool:
        return self.lstm.bidirectional

    def forward(self,
                x: Tensor,
                hidden: Tensor,
                cell: Tensor) -> tuple[tuple[Tensor, Tensor], Tensor]:
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        if self.is_bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        else:
            hidden = hidden[-1, :, :]
        logits = self.linear(hidden).squeeze()
        return (hidden, cell), logits

    def init_hidden_cell(self, batch_size: int) -> tuple[Tensor, Tensor]:
        hidden = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
        return hidden, cell

    def save(self,
             model_dir: str | bytes | os.PathLike[str] | os.PathLike[bytes],
             model_name: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))

    def load(self,
             model_dir: str | bytes | os.PathLike[str] | os.PathLike[bytes],
             model_name: str) -> Self:
        self.load_state_dict(torch.load(os.path.join(model_dir, f'{model_name}.pth')))
        return self


