import os
from os import PathLike

from utils_io import windows_to_wsl_path


class ShapeIt2:
    def __init__(self,
                 shapeit2_path: str | bytes | PathLike[str] | PathLike[bytes],
                 use_wsl: bool = False):
        if not os.path.exists(shapeit2_path):
            raise FileNotFoundError(f"ShapeIt2 not found at {shapeit2_path}")
        if not os.access(shapeit2_path, os.X_OK):
            raise PermissionError(f"ShapeIt2 not executable at {shapeit2_path}")
        if not os.path.isfile(shapeit2_path):
            raise IsADirectoryError(f"ShapeIt2 is a directory at {shapeit2_path}")
        shapeit2_path = os.path.abspath(shapeit2_path)
        self.command = ""
        if use_wsl:
            self.command += "wsl "
            shapeit2_path = windows_to_wsl_path(shapeit2_path)
        self.command += shapeit2_path
        self.args = []

    def phase(self,
              ):