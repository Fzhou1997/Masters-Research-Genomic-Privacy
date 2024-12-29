from os import PathLike
import pickle

import numpy as np
import numpy.typing as npt


def read_bitarrays(
        file_path: str | bytes | PathLike[str] | PathLike[bytes]) -> npt.NDArray[np.bool_]:
    """
    Reads a file containing pickled bitarrays and converts them into a 2D NumPy array of boolean values.

    Args:
        file_path (str | bytes | PathLike[str] | PathLike[bytes]): The path to the file containing the pickled bitarrays.

    Returns:
        npt.NDArray[np.bool_]: A 2D NumPy array where each bit in the original bitarrays is represented as a boolean value.
    """
    with open(file_path, 'rb') as file:
        bitarrays = pickle.load(file)
    length = len(bitarrays[0])
    bitarrays = np.array(bitarrays, dtype=np.uint8)
    bitarrays = np.unpackbits(bitarrays, axis=1).astype(bool)
    bitarrays = bitarrays[:, :length]
    return bitarrays