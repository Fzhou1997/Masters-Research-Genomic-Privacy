import torch
from torch import Tensor, nn


class HiddenCellLSTM(nn.Module):
    """
    The HiddenCellLSTM class is a custom neural network module which encapsulates a Long Short-Term Memory (LSTM)
    network with additional functionalities. It supports both uni-directional and bidirectional LSTM configurations.

    This class builds upon the PyTorch LSTM module by managing hidden and cell states explicitly,
    providing interfaces for input and output dimensions, and organizing forward propagation
    through sequential or bidirectional iteration. It serves to customize the LSTM behavior for specific
    needs in sequential data processing, with options for seamless integration into broader neural
    network architectures via PyTorch's nn.Module interface.

    Attributes:
        _bidirectional: Indicates whether the LSTM is bidirectional.
        _lstm_forward: An instance of a uni-directional LSTM for forward sequence processing.
        _lstm_backward: An instance of a uni-directional LSTM for backward sequence processing when bidirectional.
        _batch_first: If True, the input and output tensors have batch size as the first dimension.
        _x_batch_dim: Index of the batch dimension in input.
        _x_sequence_dim: Index of the sequence dimension in input.
        _x_feature_dim: Index of the feature dimension in input.
        _hx_direction_dim: Index of the direction dimension in hidden state.
        _hx_batch_dim: Index of the batch dimension in hidden state.
        _hx_hidden_dim: Index of the hidden dimension in hidden state.
        _y_batch_dim: Index of the batch dimension in output.
        _y_sequence_dim: Index of the sequence dimension in output.
        _y_feature_dim: Index of the feature dimension in output.
        _hy_direction_dim: Index of the direction dimension in cell state.
        _hy_batch_dim: Index of the batch dimension in cell state.
        _hy_hidden_dim: Index of the hidden dimension in cell state.

    Methods:
        __init__: Initializes the HiddenCellLSTM module.
        input_size (property): Returns the input size of the LSTM module.
        hidden_size (property): Returns the hidden size of the LSTM module.
        proj_size (property): Returns the projection size of the LSTM module.
        output_size (property): Returns the output size based on projection size.
        bidirectional (property): Checks if LSTM is bidirectional.
        num_directions (property): Returns number of directions in LSTM.
        num_layers (property): Returns number of layers in LSTM.
        batch_first (property): Returns whether batch comes first in input.
        device (property): Returns the device where tensors are placed.
        dtype (property): Returns the dtype of the tensors.
        get_hx_size: Calculates sizes for initial hidden and cell states based on batch size.
        get_hx: Returns initial hidden and cell states initialized to zeros.
        check_hx: Validates the dimensions of hidden and cell states.
        check_x: Validates dimensions and feature size of the input tensor.
        forward: Computes the sequence of outputs for the given input and hidden states.
    """

    _bidirectional: bool
    _lstm_forward: torch.nn.LSTM
    _lstm_backward: torch.nn.LSTM | None

    _batch_first: bool
    _x_batch_dim: int
    _x_sequence_dim: int
    _x_feature_dim: int
    _hx_direction_dim: int
    _hx_batch_dim: int
    _hx_hidden_dim: int
    _y_batch_dim: int
    _y_sequence_dim: int
    _y_feature_dim: int
    _hy_direction_dim: int
    _hy_batch_dim: int
    _hy_hidden_dim: int

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 proj_size: int = 0,
                 bias: bool = True,
                 batch_first: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = None):
        """
        Initializes the HiddenCellLSTM module.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int): The number of features in the hidden state.
            bidirectional (bool, optional): If True, becomes a bidirectional LSTM. Defaults to False.
            proj_size (int, optional): If > 0, will use LSTM with projections of corresponding size. Defaults to 0.
            bias (bool, optional): If False, then the layer does not use bias weights. Defaults to True.
            batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Defaults to False.
            device (torch.device, optional): The device on which to place the tensors. Defaults to None.
            dtype (torch.dtype, optional): The data type of the tensors. Defaults to None.
        """
        super(HiddenCellLSTM, self).__init__()
        self._batch_first = batch_first
        self._x_batch_dim = 0 if batch_first else 1
        self._x_sequence_dim = 1 if batch_first else 0
        self._x_feature_dim = 2
        self._hx_direction_dim = 0
        self._hx_batch_dim = 1
        self._hx_hidden_dim = 2
        self._y_batch_dim = 0 if batch_first else 1
        self._y_sequence_dim = 1 if batch_first else 0
        self._y_feature_dim = 2
        self._hy_direction_dim = 0
        self._hy_batch_dim = 1
        self._hy_hidden_dim = 2
        self._bidirectional = bidirectional
        self._lstm_forward = torch.nn.LSTM(input_size=input_size,
                                           hidden_size=hidden_size,
                                           proj_size=proj_size,
                                           bidirectional=False,
                                           num_layers=1,
                                           dropout=0.0,
                                           bias=bias,
                                           batch_first=batch_first,
                                           device=device,
                                           dtype=dtype)
        self._lstm_backward = torch.nn.LSTM(input_size=input_size,
                                            hidden_size=hidden_size,
                                            proj_size=proj_size,
                                            bidirectional=False,
                                            num_layers=1,
                                            dropout=0.0,
                                            bias=bias,
                                            batch_first=batch_first,
                                            device=device,
                                            dtype=dtype) if bidirectional else None


    @property
    def input_size(self) -> int:
        """
        Returns the number of expected features in the input.
        """
        return self._lstm_forward.input_size

    @property
    def hidden_size(self) -> int:
        """
        Returns the number of features in the hidden state.
        """
        return self._lstm_forward.hidden_size

    @property
    def proj_size(self) -> int:
        """
        Returns the size of the projections if > 0, otherwise returns 0.
        """
        return self._lstm_forward.proj_size

    @property
    def output_size(self) -> int:
        """
        Returns the size of the output features.
        """
        return self.hidden_size if self.proj_size == 0 else self.proj_size

    @property
    def bidirectional(self) -> bool:
        """
        Returns True if the LSTM is bidirectional, otherwise False.
        """
        return self._bidirectional

    @property
    def num_directions(self) -> int:
        """
        Returns the number of directions (2 if bidirectional, otherwise 1).
        """
        return 2 if self.bidirectional else 1

    @property
    def num_layers(self) -> int:
        """
        Returns the number of layers in the LSTM (always 1 in this implementation).
        """
        return 1

    @property
    def batch_first(self) -> bool:
        """
        Returns True if the input and output tensors are provided as (batch, seq, feature), otherwise False.
        """
        return self._batch_first

    @property
    def device(self) -> torch.device:
        """
        Returns the device on which the tensors are placed.
        """
        return self._lstm_forward.device

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the tensors.
        """
        return self._lstm_forward.dtype

    def get_hx_size(self, batch_size: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """
        Returns the size of the hidden and cell states.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            tuple: A tuple containing the sizes of the hidden and cell states.
        """
        h_0_size = self.num_directions, batch_size, self.hidden_size if self.proj_size == 0 else self.proj_size
        c_0_size = self.num_directions, batch_size, self.hidden_size
        return h_0_size, c_0_size

    def get_hx(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """
        Returns the initial hidden and cell states filled with zeros.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            tuple: A tuple containing the initial hidden and cell states.
        """
        device = next(self._lstm_forward.parameters()).device
        dtype = next(self._lstm_forward.parameters()).dtype
        h_0_size, c_0_size = self.get_hx_size(batch_size)
        h_0 = torch.zeros(h_0_size, device=device, dtype=dtype)
        c_0 = torch.zeros(c_0_size, device=device, dtype=dtype)
        return h_0, c_0

    def check_hx(self, hx: tuple[Tensor, Tensor], batch_size: int) -> None:
        """
        Checks if the hidden and cell states have the correct size.

        Args:
            hx (tuple): A tuple containing the hidden and cell states.
            batch_size (int): The size of the batch.

        Raises:
            ValueError: If the hidden or cell states do not have the correct size.
        """
        h_0, c_0 = hx
        h_0_size, c_0_size = self.get_hx_size(batch_size)
        if h_0.size() != h_0_size:
            raise ValueError(f"Expected hidden state size {h_0_size}, got {h_0.size()}")
        if c_0.size() != c_0_size:
            raise ValueError(f"Expected cell state size {c_0_size}, got {c_0.size()}")

    def forward(self,
                x: Tensor,
                hx: tuple[Tensor, Tensor] = None) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """
        Performs the forward pass of the LSTM.

        Args:
            x (Tensor): The input tensor.
            hx (tuple, optional): A tuple containing the initial hidden and cell states. Defaults to None.

        Returns:
            tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]: A tuple containing the output hidden and cell states.
        """
        if x.dim() != 3 and x.dim() != 2:
            raise ValueError(f"Expected input tensor of size 3 or 2, got {x.dim()}")
        if x.dim() == 2:
            x = x.unsqueeze(self._x_batch_dim)
        if x.size(self._x_feature_dim) != self.input_size:
            raise ValueError(f"Expected input feature size {self.input_size}, got {x.size(self._x_feature_dim)}")
        batch_size = x.size(self._x_batch_dim)
        seq_size = x.size(self._x_sequence_dim)

        if hx is None:
            hx = self.get_hx(batch_size)
        self.check_hx(hx, batch_size)
        h_0, c_0 = hx

        hy_forward, cy_forward = [], []
        h_i_forward = h_0.index_select(self._hx_direction_dim, torch.tensor(0))
        c_i_forward = c_0.index_select(self._hx_direction_dim, torch.tensor(0))
        for i in range(seq_size):
            x_i_forward = x.index_select(self._x_sequence_dim, torch.tensor(i))
            _, (h_i_forward, c_i_forward) = self._lstm_forward(x_i_forward, (h_i_forward, c_i_forward))
            hy_forward.append(h_i_forward)
            cy_forward.append(c_i_forward)

        if not self.bidirectional:
            y_hidden = torch.stack(hy_forward, dim=self._y_sequence_dim)
            y_cell = torch.stack(cy_forward, dim=self._y_sequence_dim)
            last_hidden = h_i_forward
            last_cell = c_i_forward
            return (y_hidden, y_cell), (last_hidden, last_cell)

        hy_backward, cy_backward = [], []
        h_i_backward = h_0.index_select(self._hx_direction_dim, torch.tensor(1))
        c_i_backward = c_0.index_select(self._hx_direction_dim, torch.tensor(1))
        for i in reversed(range(seq_size)):
            x_i_backward = x.index_select(self._x_sequence_dim, torch.tensor(i))
            _, (h_i_backward, c_i_backward) = self._lstm_backward(x_i_backward, (h_i_backward, c_i_backward))
            hy_backward.append(h_i_backward)
            cy_backward.append(c_i_backward)

        y_hidden_forward = torch.stack(hy_forward, dim=self._y_sequence_dim)
        y_hidden_backward = torch.stack(hy_backward, dim=self._y_sequence_dim)
        y_hidden = torch.cat((y_hidden_forward, y_hidden_backward), dim=self._y_feature_dim)
        y_cell_forward = torch.stack(cy_forward, dim=self._y_sequence_dim)
        y_cell_backward = torch.stack(cy_backward, dim=self._y_sequence_dim)
        y_cell = torch.cat((y_cell_forward, y_cell_backward), dim=self._y_feature_dim)
        last_hidden = torch.cat((h_i_forward, h_i_backward), dim=self._hy_direction_dim)
        last_cell = torch.cat((c_i_forward, c_i_backward), dim=self._hy_direction_dim)
        return (y_hidden, y_cell), (last_hidden, last_cell)
