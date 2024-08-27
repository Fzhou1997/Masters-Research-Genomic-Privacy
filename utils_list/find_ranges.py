from itertools import groupby


def find_ranges(ints: list[int]) -> list[str]:
    """
    Finds consecutive ranges in a sorted list of integers and returns them as a list of strings.

    Args:
        ints (list[int]): A sorted list of integers.

    Returns:
        list[str]: A list of strings representing consecutive ranges.
    """
    ranges = []
    for k, g in groupby(enumerate(ints), lambda x: x[1] - x[0]):
        group = list(g)
        if len(group) == 1:
            ranges.append(str(group[0][1]))
        else:
            ranges.append(f"{group[0][1]}-{group[-1][1]}")
    return ranges
