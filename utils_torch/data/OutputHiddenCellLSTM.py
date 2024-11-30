from typing import Literal

from torch import Tensor


class OutputHiddenCellLSTM:

    _y_hidden: Tensor
    _y_cell: Tensor
    _last_hidden: Tensor
    _last_cell: Tensor

    _batched: bool
    _batch_first: bool

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

        self._batch_first = batch_first
        self._batch_dim = 0 if self._batch_first else 1
        self._seq_dim = 1 if self._batch_first else 0
        self._output_dim = 2

        assert y_hidden.dim() == 2 or y_hidden.dim() == 3, \
            f"y_hidden must have 2 or 3 dimensions, but has {y_hidden.dim()} dimensions"
        assert y_cell.dim() == 2 or y_cell.dim() == 3, \
            f"y_cell must have 2 or 3 dimensions, but has {y_cell.dim()} dimensions"
        assert last_hidden.dim() == 2 or last_hidden.dim() == 3, \
            f"last_hidden must have 2 or 3 dimensions, but has {last_hidden.dim()} dimensions"
        assert last_cell.dim() == 2 or last_cell.dim() == 3, \
            f"last_cell must have 2 or 3 dimensions, but has {last_cell.dim()} dimensions"
        assert y_hidden.dim() == y_cell.dim(), \
            f"y_hidden and y_cell must have the same number of dimensions, but have {y_hidden.dim()} and {y_cell.dim()} dimensions"
        batched = y_hidden.dim() == 3
        if not batched:
            y_hidden = y_hidden.unsqueeze(self._batch_dim)
            y_cell = y_cell.unsqueeze(self._batch_dim)

        self._num_layers = 1
        self._batch_size = y_hidden.size(self._batch_dim)
        self._seq_size = y_hidden.size(self._seq_dim)

        self._num_directions = last_hidden.size(0)
        self._bidirectional = last_hidden.size(0) > 1

        self._output_size = y_hidden.size(self._output_dim) // self._num_directions
        self._hidden_size = y_cell.size(self._output_dim) // self._num_directions
        self._proj_size = self._output_size if self._output_size != self._hidden_size else 0

        assert y_hidden.size(-1) == self._num_directions * self._output_size, \
            f"y_hidden has last dimension {y_hidden.size(-1)}, but expected {self._num_directions * self._output_size}"
        assert y_cell.size(-1) == self._num_directions * self._hidden_size, \
            f"y_cell has last dimension {y_cell.size(-1)}, but expected {self._num_directions * self._hidden_size}"
        if self._batched and self._batch_first:
            assert y_hidden.size(1) == y_cell.size(1), \
                f"The sequence dimension of y_hidden ({y_hidden.size(0)}) and y_cell ({y_cell.size(0)}) must be equal"
        else:
            assert y_hidden.size(0) == y_cell.size(0), \
                f"The sequence dimension of y_hidden ({y_hidden.size(0)}) and y_cell ({y_cell.size(0)}) must be equal"
        if self._batched and self._batch_first:
            assert y_hidden.size(0) == y_cell.size(0), \
                f"The batch dimension of y_hidden ({y_hidden.size(1)}) and y_cell ({y_cell.size(1)}) must be equal"
        elif self._batched:
            assert y_hidden.size(1) == y_cell.size(1), \
                f"The batch dimension of y_hidden ({y_hidden.size(1)}) and y_cell ({y_cell.size(1)}) must be equal"
        assert last_hidden.size(0) == last_cell.size(0), \
            f"First dimension of last_hidden and last_cell must be equal, but are {last_hidden.size(0)} and {last_cell.size(0)}"
        if self._batched:
            assert last_hidden.size(1) == last_cell.size(1), \
                f"Second dimension of last_hidden and last_cell must be equal, but are {last_hidden.size(1)} and {last_cell.size(1)}"

        self._y_hidden = y_hidden
        self._y_cell = y_cell
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

