from torch import Tensor


class OutputMultiLayerHiddenCellLSTM:

    def __init__(self,
                 y_hidden: tuple[Tensor, ...],
                 y_cell: tuple[Tensor, ...],
                 last_hidden: tuple[Tensor, ...],
                 last_cell: tuple[Tensor, ...],
                 batch_first: bool = False) -> None:
