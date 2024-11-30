from torch import Tensor


class OutputMultiLayerLSTM:

    _y_hidden: Tensor
    _last_hidden: tuple[Tensor, ...]
    _last_cell: tuple[Tensor, ...]

    _batched: bool
    _batch_first: bool

    _batch_size: int
    _seq_size_last: int
    _hidden_size: tuple[int, ...]
    _proj_size: tuple[int, ...]
    _output_size: tuple[int, ...]

    _num_layers: int
    _num_directions: tuple[int, ...]
    _bidirectional: tuple[bool, ...]


    def __init__(self,
                 y_hidden: Tensor,
                 last_hidden: tuple[Tensor, ...],
                 last_cell: tuple[Tensor, ...],
                 batch_first: bool = False) -> None:
        assert y_hidden.dim() == 2 or y_hidden.dim() == 3, \
            f"y_hidden must have 2 or 3 dimensions, but has {y_hidden.dim()} dimensions"
        self._batched = y_hidden.dim() == 3
        self._batch_first = batch_first

