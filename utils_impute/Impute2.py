import os
from os import PathLike


class Impute2:
    def __init__(self,
                 impute2_path: str | bytes | PathLike[str] | PathLike[bytes],
                 is_wsl: bool = False):
        self.command = ""
        if is_wsl:
            self.command += "wsl "
        if not os.path.exists(impute2_path):
            raise FileNotFoundError(f"Impute2 not found at {impute2_path}")
        if not os.access(impute2_path, os.X_OK):
            raise PermissionError(f"Impute2 not executable at {impute2_path}")
        if not os.path.isabs(impute2_path):
            impute2_path = os.path.abspath(impute2_path)

        self.command += str(impute2_path)
