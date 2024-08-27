import re

from .extract_integers import extract_integers
from .find_ranges import find_ranges


def create_summary(strs: list[str]) -> str:
    """
    Creates a summary string from a list of strings by extracting integers,
    finding consecutive ranges, and appending non-integer strings.

    Args:
        strs (list[str]): A list of strings to process.

    Returns:
        str: A summary string containing ranges of integers and other strings.
    """
    ints = extract_integers(strs)
    ranges = find_ranges(ints)
    other_strs = [s for s in strs if not re.search(r'\d+', s)]
    if 'X' in other_strs:
        ranges.append('X')
    if 'Y' in other_strs:
        ranges.append('Y')
    if 'MT' in other_strs:
        ranges.append('MT')
    summary = ', '.join(ranges)
    return summary
