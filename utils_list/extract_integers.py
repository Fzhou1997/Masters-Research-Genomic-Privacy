import re


def extract_integers(strs: list[str]) -> list[int]:
    """
    Extracts integers from a list of strings and returns them as a sorted list.

    Args:
        strs (list[str]): A list of strings to extract integers from.

    Returns:
        list[int]: A sorted list of integers extracted from the input strings.
    """
    return sorted([int(re.search(r'\d+', s).group()) for s in strs if re.search(r'\d+', s)])
