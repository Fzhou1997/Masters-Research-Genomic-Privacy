from torch import jit


class LSTMLayerHiddenCell(jit.ScriptModule):

    _bidirectional: bool

    _lstm_forward:


    def __init__(self):
