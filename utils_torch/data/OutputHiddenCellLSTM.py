from typing import Literal

import torch
from torch import Tensor


class OutputHiddenCellLSTM:
    """
    A class to handle the output of an HiddenCellLSTM.

    Attributes:
    -----------
    _y_hidden : Tensor
        The hidden states of the LSTM.
    _y_cell : Tensor
        The cell states of the LSTM.
    _last_hidden : Tensor
        The last hidden states of the LSTM.
    _last_cell : Tensor
        The last cell states of the LSTM.
    _y_batch_dim : int
        The batch dimension index for the output hidden and cell states.
    _y_seq_dim : int
        The sequence dimension index for the output hidden and cell states.
    _y_hidden_dim : int
        The hidden dimension index for the output hidden and cell states.
    _last_direction_dim : int
        The direction dimension index for the last hidden and cell states.
    _last_batch_dim : int
        The batch dimension index for the last hidden and cell states.
    _last_hidden_dim : int
        The hidden dimension index for the last hidden and cell states.
    _bidirectional : bool
        Indicates if the LSTM is bidirectional.
    _num_directions : int
        The number of directions (1 for unidirectional, 2 for bidirectional).
    _num_layers : int
        The number of layers in the LSTM (always 1 in this class).
    _batch_size : int
        The batch size of the input.
    _seq_size : int
        The sequence length of the input.
    _hidden_size : int
        The hidden size of the LSTM.
    _proj_size : int
        The projection size of the LSTM.
    _output_size : int
        The output size of the LSTM.
    """

    _y_hidden: Tensor
    _y_cell: Tensor
    _last_hidden: Tensor
    _last_cell: Tensor

    _y_batch_dim: int
    _y_seq_dim: int
    _y_hidden_dim: int
    _last_direction_dim: int
    _last_batch_dim: int
    _last_hidden_dim: int

    _bidirectional: bool
    _num_directions: int
    _num_layers: int

    _batch_size: int
    _seq_size: int
    _hidden_size: int
    _proj_size: int
    _output_size: int

    def __init__(self,
                 y_hidden: Tensor,
                 y_cell: Tensor,
                 last_hidden: Tensor,
                 last_cell: Tensor,
                 batch_first: bool = False) -> None:
        """
        Initializes the OutputHiddenCellLSTM class.

        Parameters:
        -----------
        y_hidden : Tensor
            The hidden states of the LSTM.
        y_cell : Tensor
            The cell states of the LSTM.
        last_hidden : Tensor
            The last hidden states of the LSTM.
        last_cell : Tensor
            The last cell states of the LSTM.
        batch_first : bool, optional
            If True, the input and output tensors are provided as (batch, seq, feature).
            Default is False.
        """
        if not batch_first:
            y_hidden = y_hidden.transpose(0, 1)
            y_cell = y_cell.transpose(0, 1)

        self._y_batch_dim = 0
        self._y_seq_dim = 1
        self._y_hidden_dim = 2
        self._last_direction_dim = 0
        self._last_batch_dim = 1
        self._last_hidden_dim = 2

        assert y_hidden.dim() == 2 or y_hidden.dim() == 3, \
            f"y_hidden must have 2 or 3 dimensions, but has {y_hidden.dim()} dimensions"
        batched = y_hidden.dim() == 3
        assert y_hidden.dim() == y_cell.dim() == last_hidden.dim() == last_cell.dim(), \
            f"y_hidden, y_cell, last_hidden, and last_cell must have the same number of dimensions, " \
            f"but are {y_hidden.dim()}, {y_cell.dim()}, {last_hidden.dim()}, and {last_cell.dim()}"
        if not batched:
            y_hidden = y_hidden.unsqueeze(self._y_batch_dim)
            y_cell = y_cell.unsqueeze(self._y_batch_dim)
            last_hidden = last_hidden.unsqueeze(self._last_batch_dim)
            last_cell = last_cell.unsqueeze(self._last_batch_dim)
        assert (y_hidden.size(self._y_batch_dim)
                == y_cell.size(self._y_batch_dim)
                == last_hidden.size(self._last_batch_dim)
                == last_cell.size(self._last_batch_dim)), \
            f"Batch dimension of y_hidden ({y_hidden.size(0)}), y_cell ({y_cell.size(0)}), " \
            f"last_hidden ({last_hidden.size(0)}), and last_cell ({last_cell.size(0)}) must be equal"

        self._num_layers = 1
        self._batch_size = y_hidden.size(self._y_batch_dim)
        self._seq_size = y_hidden.size(self._y_seq_dim)

        self._num_directions = last_hidden.size(self._last_direction_dim)
        self._bidirectional = last_hidden.size(self._last_direction_dim) > 1

        self._output_size = y_hidden.size(self._y_hidden_dim) // self._num_directions
        self._hidden_size = y_cell.size(self._y_hidden_dim) // self._num_directions
        self._proj_size = self._output_size if self._output_size != self._hidden_size else 0

        assert y_hidden.size(self._y_seq_dim) == y_cell.size(self._y_seq_dim), \
            f"Sequence dimension of y_hidden ({y_hidden.size(1)}) and y_cell ({y_cell.size(1)}) must be equal"
        assert y_hidden.size(self._y_hidden_dim) == self._num_directions * self._output_size, \
            f"y_hidden has last dimension {y_hidden.size(-1)}, but expected {self._num_directions * self._output_size}"
        assert y_cell.size(self._y_hidden_dim) == self._num_directions * self._hidden_size, \
            f"y_cell has last dimension {y_cell.size(-1)}, but expected {self._num_directions * self._hidden_size}"

        assert last_hidden.size(self._last_direction_dim) == last_cell.size(
            self._last_direction_dim), \
            f"First dimension of last_hidden and last_cell must be equal, but are {last_hidden.size(0)} and {last_cell.size(0)}"

        self._y_hidden = y_hidden
        self._y_cell = y_cell
        self._last_hidden = last_hidden
        self._last_cell = last_cell

    @property
    def bidirectional(self) -> bool:
        """
        Returns whether the LSTM is bidirectional.
        """
        return self._bidirectional

    @property
    def num_directions(self) -> int:
        """
        Returns the number of directions (1 for unidirectional, 2 for bidirectional).
        """
        return self._num_directions

    @property
    def num_layers(self) -> int:
        """
        Returns the number of layers in the LSTM (always 1 in this class).
        """
        return self._num_layers

    @property
    def batch_size(self) -> int:
        """
        Returns the batch size of the input.
        """
        return self._batch_size

    @property
    def seq_size(self) -> int:
        """
        Returns the sequence length of the input.
        """
        return self._seq_size

    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the LSTM.
        """
        return self._hidden_size

    @property
    def proj_size(self) -> int:
        """
        Returns the projection size of the LSTM.
        """
        return self._proj_size

    @property
    def output_size(self) -> int:
        """
        Returns the output size of the LSTM.
        """
        return self._output_size

    def y_hidden(self,
                 batch_idx: int | slice = slice(None),
                 seq_idx: int | slice = slice(None),
                 output_idx: int | slice = slice(None),
                 direction: Literal["forward", "backward", "both"] = "both") -> Tensor:
        """
        Returns the hidden states of the LSTM.

        Parameters:
        -----------
        batch_idx : int or slice, optional
            The batch index or slice. Default is slice(None).
        seq_idx : int or slice, optional
            The sequence index or slice. Default is slice(None).
        output_idx : int or slice, optional
            The output index or slice. Default is slice(None).
        direction : Literal["forward", "backward", "both"], optional
            The direction of the hidden states to return. Default is "both".

        Returns:
        --------
        Tensor
            The hidden states of the LSTM.
        """
        if self._num_directions == 1:
            direction = "forward"
        if direction == "forward":
            return self._y_hidden[batch_idx, seq_idx, output_idx]
        elif direction == "backward":
            return self._y_hidden[batch_idx, seq_idx, self._output_size + output_idx]
        else:
            forward = self._y_hidden[batch_idx, seq_idx, output_idx]
            backward = self._y_hidden[batch_idx, seq_idx, self._output_size + output_idx]
            return torch.cat((forward, backward), dim=-1)

    def y_cell(self,
               batch_idx: int | slice = slice(None),
               seq_idx: int | slice = slice(None),
               hidden_idx: int | slice = slice(None),
               direction: Literal["forward", "backward", "both"] = "both") -> Tensor:
        """
        Returns the cell states of the LSTM.

        Parameters:
        -----------
        batch_idx : int or slice, optional
            The batch index or slice. Default is slice(None).
        seq_idx : int or slice, optional
            The sequence index or slice. Default is slice(None).
        hidden_idx : int or slice, optional
            The hidden index or slice. Default is slice(None).
        direction : Literal["forward", "backward", "both"], optional
            The direction of the cell states to return. Default is "both".

        Returns:
        --------
        Tensor
            The cell states of the LSTM.
        """
        if self._num_directions == 1:
            direction = "forward"
        if direction == "forward":
            return self._y_cell[batch_idx, seq_idx, hidden_idx]
        elif direction == "backward":
            return self._y_cell[batch_idx, seq_idx, self._hidden_size + hidden_idx]
        else:
            forward = self._y_cell[batch_idx, seq_idx, hidden_idx]
            backward = self._y_cell[batch_idx, seq_idx, self._hidden_size + hidden_idx]
            return torch.cat((forward, backward), dim=-1)

    def last_hidden(self,
                    batch_idx: int | slice = slice(None),
                    output_idx: int | slice = slice(None),
                    direction: Literal["forward", "backward", "both"] = "both") -> Tensor:
        """
        Returns the last hidden states of the LSTM.

        Parameters:
        -----------
        batch_idx : int or slice, optional
            The batch index or slice. Default is slice(None).
        output_idx : int or slice, optional
            The output index or slice. Default is slice(None).
        direction : Literal["forward", "backward", "both"], optional
            The direction of the last hidden states to return. Default is "both".

        Returns:
        --------
        Tensor
            The last hidden states of the LSTM.
        """
        if self._num_directions == 1:
            direction = "forward"
        if direction == "forward":
            return self._last_hidden[0, batch_idx, output_idx]
        elif direction == "backward":
            return self._last_hidden[1, batch_idx, output_idx]
        else:
            return self._last_hidden[:, batch_idx, output_idx]

    def last_cell(self,
                  batch_idx: int | slice = slice(None),
                  hidden_idx: int | slice = slice(None),
                  direction: Literal["forward", "backward", "both"] = "both") -> Tensor:
        """
        Returns the last cell states of the LSTM.

        Parameters:
        -----------
        batch_idx : int or slice, optional
            The batch index or slice. Default is slice(None).
        hidden_idx : int or slice, optional
            The hidden index or slice. Default is slice(None).
        direction : Literal["forward", "backward", "both"], optional
            The direction of the last cell states to return. Default is "both".

        Returns:
        --------
        Tensor
            The last cell states of the LSTM.
        """
        if self._num_directions == 1:
            direction = "forward"
        if direction == "forward":
            return self._last_cell[0, batch_idx, hidden_idx]
        elif direction == "backward":
            return self._last_cell[1, batch_idx, hidden_idx]
        else:
            return self._last_cell[:, batch_idx, hidden_idx]
