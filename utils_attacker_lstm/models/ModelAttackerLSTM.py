import os
from os import PathLike
from typing import Self, Sequence
from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from utils_torch.modules import MultiLayerLSTM, hx_type


class ModelAttackerLSTM(nn.Module):
    """
    A PyTorch module for an LSTM-based attacker model.

    Attributes:
        lstm_modules (MultiLayerLSTM): The LSTM layers used in the model.
    """

    lstm_modules: MultiLayerLSTM

    def __init__(self,
                 lstm_num_layers: int,
                 lstm_input_size: int,
                 lstm_hidden_size: int | Sequence[int],
                 lstm_proj_size: int | Sequence[int] = 0,
                 lstm_bidirectional: bool | Sequence[bool] = False,
                 lstm_dropout_p: float | Sequence[float] = 0.5,
                 lstm_dropout_first: bool | Sequence[bool] = True,
                 lstm_layer_norm: bool | Sequence[bool] = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None) -> None:
        """
        Initializes the ModelAttackerLSTM.

        Args:
            lstm_num_layers (int): The number of layers in the LSTM.
            lstm_input_size (int): The input size of the LSTM.
            lstm_hidden_size (int | Sequence[int]): The hidden size of the LSTM.
            lstm_proj_size (int | Sequence[int], optional): The projection size of the LSTM. Defaults to 0.
            lstm_bidirectional (bool | Sequence[bool], optional): Whether the LSTM is bidirectional. Defaults to False.
            lstm_dropout_p (float | Sequence[float], optional): The dropout rate of the LSTM. Defaults to 0.5.
            lstm_dropout_first (bool | Sequence[bool], optional): Whether the dropout is applied to the input. Defaults to True.
            lstm_layer_norm (bool | Sequence[bool], optional): Whether layer normalization is applied. Defaults to True.
            device (torch.device, optional): Device for the tensors. Defaults to None.
            dtype (torch.dtype, optional): Data type for the tensors. Defaults to None.
        """
        super(ModelAttackerLSTM, self).__init__()
        self.lstm_modules = MultiLayerLSTM(num_layers=lstm_num_layers,
                                           input_size=lstm_input_size,
                                           hidden_size=lstm_hidden_size,
                                           proj_size=lstm_proj_size,
                                           bidirectional=lstm_bidirectional,
                                           dropout_p=lstm_dropout_p,
                                           dropout_first=lstm_dropout_first,
                                           layer_norm=lstm_layer_norm,
                                           device=device,
                                           dtype=dtype)

    @abstractmethod
    def forward(self,
                x: Tensor,
                hx: hx_type = None) -> tuple[Tensor, hx_type]:
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input tensor.
            hx (tuple[tuple[Tensor, ...], tuple[Tensor, ...]], optional): The initial hidden and cell states. Defaults to None.

        Returns:
            tuple[Tensor, tuple[tuple[Tensor, ...], tuple[Tensor, ...]]]: The output tensor and the final hidden and cell
        """
        return self.lstm_modules.forward(x, hx)

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
        return self.lstm_modules.get_hx(batch_size=batch_size, device=device, dtype=dtype)

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

    @property
    def lstm_num_layers(self) -> int:
        """
        Returns the number of layers in the LSTM.

        Returns:
            int: The number of layers in the LSTM.
        """
        return self.lstm_modules.lstm_num_layers

    @property
    def lstm_input_size_in(self) -> int:
        """
        Returns the input size of the LSTM.

        Returns:
            int: The input size of the LSTM.
        """
        return self.lstm_modules.lstm_input_size_in

    @property
    def lstm_output_size_out(self) -> int:
        """
        Returns the output size of the LSTM.

        Returns:
            int: The output size of the LSTM.
        """
        return self.lstm_modules.lstm_output_size_out

    @property
    def lstm_input_size(self) -> tuple[int, ...]:
        """
        Returns the input size of the LSTM.

        Returns:
            tuple[int, ...]: The input size of the LSTM.
        """
        return self.lstm_modules.lstm_input_size

    @property
    def lstm_hidden_size(self) -> tuple[int, ...]:
        """
        Returns the hidden size of the LSTM.

        Returns:
            tuple[int, ...]: The hidden size of the LSTM.
        """
        return self.lstm_modules.lstm_hidden_size

    @property
    def lstm_proj_size(self) -> tuple[int, ...]:
        """
        Returns the projection size of the LSTM.

        Returns:
            tuple[int, ...]: The projection size of the LSTM.
        """
        return self.lstm_modules.lstm_proj_size

    @property
    def lstm_output_size(self) -> tuple[int, ...]:
        """
        Returns the output size of the LSTM.

        Returns:
            tuple[int, ...]: The output size of the LSTM.
        """
        return self.lstm_modules.lstm_output_size

    @property
    def lstm_bidirectional(self) -> tuple[bool, ...]:
        """
        Returns whether the LSTM is bidirectional.

        Returns:
            tuple[bool, ...]: True if the LSTM is bidirectional, False otherwise.
        """
        return self.lstm_modules.lstm_bidirectional

    @property
    def lstm_num_directions(self) -> tuple[int, ...]:
        """
        Returns the number of directions in the LSTM (1 for unidirectional, 2 for bidirectional).

        Returns:
            tuple[int, ...]: The number of directions in the LSTM.
        """
        return self.lstm_modules.lstm_num_directions


    @property
    def lstm_dropout_p(self) -> tuple[float, ...]:
        """
        Returns the dropout_p rate of the LSTM.

        Returns:
            tuple[float, ...]: The dropout_p rate of the LSTM.
        """
        return self.lstm_modules.lstm_dropout_p

    @property
    def lstm_dropout_first(self) -> tuple[bool, ...]:
        """
        Returns the dropout_first rate of the LSTM.

        Returns:
            tuple[bool, ...]: The dropout_first rate of the LSTM.
        """
        return self.lstm_modules.lstm_dropout_first

    @property
    def lstm_layer_norm(self) -> tuple[bool, ...]:
        """
        Returns the layer_norm rate of the LSTM.

        Returns:
            tuple[bool, ...]: The layer_norm rate of the LSTM.
        """
        return self.lstm_modules.lstm_layer_norm
