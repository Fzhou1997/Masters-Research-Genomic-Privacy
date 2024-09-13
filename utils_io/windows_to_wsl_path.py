import os
from os import PathLike


def windows_to_wsl_path(
        windows_path: str | bytes | PathLike[str] | PathLike[bytes]) -> str:
    """
    Convert a Windows file path to a WSL (Windows Subsystem for Linux) file path.

    Parameters:
    windows_path (str | bytes | PathLike[str] | PathLike[bytes]): The Windows file path to convert.

    Returns:
    str: The corresponding WSL file path.
    """
    windows_path = os.path.abspath(windows_path)
    unix_path = windows_path.replace("\\", "/")
    wsl_path = f"/mnt/{unix_path[0].lower()}/{unix_path[3:]}"
    if " " in wsl_path:
        wsl_path = f"'{wsl_path}'"
    return wsl_path
