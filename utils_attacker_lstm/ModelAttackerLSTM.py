import os
from os import PathLike
from typing import Self
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class ModelAttackerLSTM(nn.Module):

    lstm: nn.LSTM

    def __init__(self,
                 lstm_input_size: int,
                 lstm_hidden_size: int,
                 lstm_num_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 lstm_dropout: float = 0.5):
        super(ModelAttackerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True,
                            dropout=lstm_dropout if lstm_num_layers > 1 else 0)

    @property
    def lstm_input_size(self) -> int:
        return self.lstm.input_size

    @property
    def lstm_hidden_size(self) -> int:
        return self.lstm.hidden_size

    @property
    def lstm_num_layers(self) -> int:
        return self.lstm.num_layers

    @property
    def lstm_num_directions(self) -> int:
        return 2 if self.lstm.bidirectional else 1

    @property
    def lstm_dropout(self) -> float:
        return self.lstm.dropout

    @property
    def lstm_bidirectional(self) -> bool:
        return self.lstm.bidirectional

    @abstractmethod
    def forward(self,
                x: Tensor,
                hidden: Tensor,
                cell: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, logits: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def classify(self, predicted: Tensor) -> Tensor:
        raise NotImplementedError

    def init_hidden_cell(self, batch_size: int) -> tuple[Tensor, Tensor]:
        hidden = torch.zeros(self.lstm_num_layers * self.lstm_num_directions, batch_size, self.lstm_hidden_size)
        cell = torch.zeros(self.lstm_num_layers * self.lstm_num_directions, batch_size, self.lstm_hidden_size)
        return hidden, cell

    def save(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))

    def load(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> Self:
        self.load_state_dict(torch.load(os.path.join(model_dir, f'{model_name}.pth')))
        return self
