import os
from os import PathLike
from typing import Self
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class ModelAttackerLSTM(nn.Module):
    """
    A PyTorch module for an LSTM-based model attacker.

    Attributes:
        lstm (nn.LSTM): The LSTM layer used in the model.
    """

    lstm: nn.LSTM

    def __init__(self,
                 lstm_input_size: int,
                 lstm_hidden_size: int,
                 lstm_num_layers: int = 1,
                 lstm_bidirectional: bool = False,
                 lstm_dropout: float = 0.5):
        """
        Initializes the ModelAttackerLSTM.

        Args:
            lstm_input_size (int): The number of expected features in the input.
            lstm_hidden_size (int): The number of features in the hidden state.
            lstm_num_layers (int, optional): Number of recurrent layers. Default is 1.
            lstm_bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Default is False.
            lstm_dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer. Default is 0.5.
        """
        super(ModelAttackerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True,
                            dropout=lstm_dropout if lstm_num_layers > 1 else 0)

    @property
    def lstm_input_size(self) -> int:
        """
        Returns the input size of the LSTM.

        Returns:
            int: The input size of the LSTM.
        """
        return self.lstm.input_size

    @property
    def lstm_hidden_size(self) -> int:
        """
        Returns the hidden size of the LSTM.

        Returns:
            int: The hidden size of the LSTM.
        """
        return self.lstm.hidden_size

    @property
    def lstm_num_layers(self) -> int:
        """
        Returns the number of layers in the LSTM.

        Returns:
            int: The number of layers in the LSTM.
        """
        return self.lstm.num_layers

    @property
    def lstm_num_directions(self) -> int:
        """
        Returns the number of directions in the LSTM (1 for unidirectional, 2 for bidirectional).

        Returns:
            int: The number of directions in the LSTM.
        """
        return 2 if self.lstm.bidirectional else 1

    @property
    def lstm_dropout(self) -> float:
        """
        Returns the dropout_p rate of the LSTM.

        Returns:
            float: The dropout_p rate of the LSTM.
        """
        return self.lstm.dropout

    @property
    def lstm_bidirectional(self) -> bool:
        """
        Returns whether the LSTM is bidirectional.

        Returns:
            bool: True if the LSTM is bidirectional, False otherwise.
        """
        return self.lstm.bidirectional

    @abstractmethod
    def forward(self,
                x: Tensor,
                hidden: Tensor,
                cell: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input tensor.
            hidden (Tensor): The hidden state tensor.
            cell (Tensor): The cell state tensor.

        Returns:
            tuple[Tensor, tuple[Tensor, Tensor]]: The output tensor and the new hidden and cell states.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, logits: Tensor) -> Tensor:
        """
        Predicts the output based on the logits.

        Args:
            logits (Tensor): The input logits.

        Returns:
            Tensor: The predicted output.
        """
        raise NotImplementedError

    @abstractmethod
    def classify(self, predicted: Tensor) -> Tensor:
        """
        Classifies the predicted output.

        Args:
            predicted (Tensor): The predicted output.

        Returns:
            Tensor: The classified output.
        """
        raise NotImplementedError

    def init_hidden_cell(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """
        Initializes the hidden and cell states.

        Args:
            batch_size (int): The batch size.

        Returns:
            tuple[Tensor, Tensor]: The initialized hidden and cell states.
        """
        hidden = torch.zeros(self.lstm_num_layers * self.lstm_num_directions, batch_size, self.lstm_hidden_size)
        cell = torch.zeros(self.lstm_num_layers * self.lstm_num_directions, batch_size, self.lstm_hidden_size)
        return hidden, cell

    def save(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> None:
        """
        Saves the model state to a file.

        Args:
            model_dir (str | bytes | PathLike[str] | PathLike[bytes]): The directory to save the model.
            model_name (str): The name of the model file.
        """
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))

    def load(self,
             model_dir: str | bytes | PathLike[str] | PathLike[bytes],
             model_name: str) -> Self:
        """
        Loads the model state from a file.

        Args:
            model_dir (str | bytes | PathLike[str] | PathLike[bytes]): The directory to load the model from.
            model_name (str): The name of the model file.

        Returns:
            Self: The loaded model.
        """
        self.load_state_dict(torch.load(os.path.join(model_dir, f'{model_name}.pth')))
        return self
