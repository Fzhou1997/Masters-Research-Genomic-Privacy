from typing import Literal

import torch
from torch import Tensor


class OutputLSTM:

    _y_hidden: Tensor
    _last_hidden: Tensor
    _last_cell: Tensor

    _batched: bool
    _batch_first: bool

    _batch_size: int
    _seq_size: int
    _hidden_size: int
    _proj_size: int
    _output_size: int

    _num_layers: int
    _num_directions: int
    _bidirectional: bool

    def __init__(self,
                 y_hidden: Tensor,
                 last_hidden: Tensor,
                 last_cell: Tensor,
                 batch_first: bool = False,
                 bidirectional: bool = False) -> None:

        assert y_hidden.dim() == 2 or y_hidden.dim() == 3, \
            f"y_hidden must have 2 or 3 dimensions, but has {y_hidden.dim()} dimensions"
        self._batched = y_hidden.dim() == 3
        self._batch_first = batch_first
        self._bidirectional = bidirectional
        self._num_directions = 2 if self._bidirectional else 1

        if not self._batched:
            self._seq_size = y_hidden.size(0)
            self._batch_size = 1
        elif not self._batch_first:
            self._seq_size = y_hidden.size(0)
            self._batch_size = y_hidden.size(1)
        else:
            self._batch_size = y_hidden.size(0)
            self._seq_size = y_hidden.size(1)

        if last_hidden.size(1) == last_cell.size(1):
            self._hidden_size = last_cell.size(1)
            self._proj_size = 0
        else:
            self._hidden_size = last_cell.size(1)
            self._proj_size = last_hidden.size(1)
        self._output_size = self._hidden_size if self._proj_size == 0 else self._proj_size

        self._num_layers = last_hidden.size(0) // self._num_directions

        assert y_hidden.size(-1) == self._num_directions * self._output_size, \
            f"Last dimension of y_hidden must be {self._num_directions * self._output_size}, but is {y_hidden.size(-1)}"
        assert last_hidden.size(0) == last_cell.size(0), \
            f"First dimension of last_hidden and last_cell must be equal, but are {last_hidden.size(0)} and {last_cell.size(0)}"
        if self._batched:
            assert last_hidden.size(1) == last_cell.size(1), \
                f"Second dimension of last_hidden and last_cell must be equal, but are {last_hidden.size(1)} and {last_cell.size(1)}"

        self._y_hidden = y_hidden
        self._last_hidden = last_hidden
        self._last_cell = last_cell

    @property
    def batched(self) -> bool:
        return self._batched

    @property
    def batch_first(self) -> bool:
        return self._batch_first

    @property
    def bidirectional(self) -> bool:
        return self._bidirectional

    @property
    def num_directions(self) -> int:
        return self._num_directions

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def seq_size(self) -> int:
        return self._seq_size

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def proj_size(self) -> int:
        return self._proj_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def y_hidden(self,
                 batch_idx: int | slice = slice(None),
                 seq_idx: int | slice = slice(None),
                 output_idx: int | slice = slice(None),
                 direction: Literal["forward", "backward", "both"] = "both") -> Tensor:
        if not self._batched:
            if self._bidirectional:
                if direction == "forward":
                    return self._y_hidden[seq_idx, output_idx]
                elif direction == "backward":
                    return self._y_hidden[seq_idx, output_idx + self._output_size]
                else:
                    forward = self._y_hidden[seq_idx, output_idx]
                    backward = self._y_hidden[seq_idx, output_idx + self._output_size]
                    return torch.cat((forward, backward), dim=-1)
            else:
                return self._y_hidden[seq_idx, output_idx]
        else:
            if self._batch_first:
                if self._bidirectional:
                    if direction == "forward":
                        return self._y_hidden[batch_idx, seq_idx, output_idx]
                    elif direction == "backward":
                        return self._y_hidden[batch_idx, seq_idx, output_idx + self._output_size]
                    else:
                        forward = self._y_hidden[batch_idx, seq_idx, output_idx]
                        backward = self._y_hidden[batch_idx, seq_idx, output_idx + self._output_size]
                        return torch.cat((forward, backward), dim=-1)
                else:
                    return self._y_hidden[batch_idx, seq_idx, output_idx]
            else:
                if self._bidirectional:
                    if direction == "forward":
                        return self._y_hidden[seq_idx, batch_idx, output_idx]
                    elif direction == "backward":
                        return self._y_hidden[seq_idx, batch_idx, output_idx + self._output_size]
                    else:
                        forward = self._y_hidden[seq_idx, batch_idx, output_idx]
                        backward = self._y_hidden[seq_idx, batch_idx, output_idx + self._output_size]
                        return torch.cat((forward, backward), dim=-1)
                else:
                    return self._y_hidden[seq_idx, batch_idx, output_idx]
        
    def last_hidden(self,
                    batch_idx: int | slice = slice(None),
                    layer_idx: int | slice = slice(None),
                    output_idx: int | slice = slice(None),
                    direction: Literal["forward", "backward", "both"] = "both") -> Tensor:
        first_idx = layer_idx
        if not self._bidirectional:
            first_idx = layer_idx
        elif isinstance(layer_idx, int):
            if direction == "both":
                first_idx = slice(layer_idx * 2, layer_idx * 2 + 2, 1)
            elif direction == "forward":
                first_idx = layer_idx * 2
            elif direction == "backward":
                first_idx = layer_idx * 2 + 1
        else:
            if direction == "both":
                first_idx = slice(layer_idx.start * 2, layer_idx.stop * 2, 1)
            elif direction == "forward":
                first_idx = slice(layer_idx.start * 2, layer_idx.stop * 2, 2)
            elif direction == "backward":
                first_idx = slice(layer_idx.start * 2 + 1, layer_idx.stop * 2 + 1, 2)
        if not self._batched:
            return self._last_hidden[first_idx, output_idx]
        else:
            return self._last_hidden[first_idx, batch_idx, output_idx]

    def last_cell(self,
                  batch_idx: int | slice = slice(None),
                  layer_idx: int | slice = slice(None),
                  cell_idx: int | slice = slice(None),
                  direction: Literal["forward", "backward", "both"] = "both") -> Tensor:
            first_idx = layer_idx
            if not self._bidirectional:
                first_idx = layer_idx
            elif isinstance(layer_idx, int):
                if direction == "both":
                    first_idx = slice(layer_idx * 2, layer_idx * 2 + 2, 1)
                elif direction == "forward":
                    first_idx = layer_idx * 2
                elif direction == "backward":
                    first_idx = layer_idx * 2 + 1
            else:
                if direction == "both":
                    first_idx = slice(layer_idx.start * 2, layer_idx.stop * 2, 1)
                elif direction == "forward":
                    first_idx = slice(layer_idx.start * 2, layer_idx.stop * 2, 2)
                elif direction == "backward":
                    first_idx = slice(layer_idx.start * 2 + 1, layer_idx.stop * 2 + 1, 2)
            if not self._batched:
                return self._last_cell[first_idx, cell_idx]
            else:
                return self._last_cell[first_idx, batch_idx, cell_idx]
