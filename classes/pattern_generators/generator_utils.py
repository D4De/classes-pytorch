import math
from typing import Any, List, Sequence, Tuple
import numpy as np

import tarfile

tarfile


def random_int_from_pct_range(
    number: int, range_min_pct: float, range_max_pct: float
) -> int:
    """
    Extracts a random integer between the ``range_min_pct``% and ``range_max_pct``% of
    ``number``

    Args
    ---
    * ``number``: The base value
    * ``min_val``: The minimum percetange of ``number`` that can be extracted
    * ``max_val``: The maximum percetange of ``number`` that can be extracted

    Returns
    ---
    A random integer extracted uniformly between ``number * range_min_pct / 100.0``
    and ``number * range_max_pct / 100.0``. If these two values are not integer
    they are rounded to the nearest integer.
    """
    min_number = int(round(number * range_min_pct / 100.0))
    max_number = int(round(number * range_max_pct / 100.0))

    return np.random.randint(min_number, max_number + 1)


def clamp(val, min_val, max_val):
    """
    Restricts a number between ``min_val`` and ``max_val``

    Args
    ---
    * ``val``: The number to be clamped
    * ``min_val``: The minimum value that can be returned
    * ``max_val``: The maximum value that can be returned

    Returns
    ---
    ``max(min_val, min(val, max_val))``
    """
    return max(min_val, min(val, max_val))


def random_list_with_gap_constraints(
    length: int, max_number: int, min_gap: int, max_gap: int, strict: bool = False
) -> np.ndarray:
    """
    Extracts uniformly random ordered a list of integers containg ``length`` numbers between ``0`` and ``max_number``.
    Each number has a distance from the previous between ``min_gap`` and ``max_gap``.
    The first number is of the list always ``0`` and the last number of the list ``max_number`` (if the args allow it)

    Args
    ---
    * ``length``: The lenght of the output list
    * ``max_number``: The last number at the end of the list
    * ``min_gap``: Minimum distance between two consecutive number of the list
    * ``max_gap``: Maximum distance between two consecutive number of the list
    * ``strict``: Throws ``ValueError`` if the parameters constitute an impossible problem

    Returns
    ---
    A list of integer with length ``length``. Each couple of consecutive numbers has a distance between ``min_gap`` and ``max_gap``.
    """
    # print(f'{length=} {max_number=} {min_gap=} {max_gap=}')
    if length == 0:
        return np.array([], dtype=np.int64)
    if length == 1:
        return np.array([0], dtype=np.int64)
    if min_gap == max_gap:
        return np.arange(0, max_number, min_gap)[:length].astype(np.int64)

    gap_list = np.full((length - 1,), min_gap, dtype=np.int64)

    tail = min_gap * (length - 1)
    distance_to_max_number = max_number - tail
    if distance_to_max_number <= 0:
        if strict:
            raise ValueError(
                f"List is impossible to costruct with parameters {length=} {max_number=} {min_gap=} {max_gap=}"
            )
        increment = False
        distance_to_max_number = tail - max_number
        possible_incremenents_per_gap = min_gap - 1
        possible_incremenents = possible_incremenents_per_gap * (length - 1)
    else:
        increment = True
        possible_incremenents_per_gap = max_gap - min_gap
        possible_incremenents = possible_incremenents_per_gap * (length - 1)

    increments_ids = np.random.choice(
        possible_incremenents, size=distance_to_max_number, replace=False
    )
    gap_ids = np.intp(np.floor(increments_ids / possible_incremenents_per_gap))
    gap_ids, gap_counts = np.unique(gap_ids, return_counts=True)
    if increment:
        gap_list[gap_ids] += gap_counts
    else:
        gap_list[gap_ids] -= gap_counts

    result = np.zeros((length,), dtype=np.intp)
    np.cumsum(gap_list, out=result[1:])

    # print(result)
    return result


def random_channels(
    num_channels: int,
    min_channel_skip: int,
    max_channel_skip: int,
    max_corrupted_channels: int,
    corrupted_chan_min_pct: float,
    corrupted_chan_max_pct: float,
    min_channels: int = 1,
) -> List[int]:

    max_channels_for_gaps = int(math.floor(num_channels / min_channel_skip)) + 1
    num_corrupted_channels = random_int_from_pct_range(
        num_channels, corrupted_chan_min_pct, corrupted_chan_max_pct
    )

    num_corrupted_channels = max(
        min(num_corrupted_channels, max_channels_for_gaps, max_corrupted_channels),
        min(num_channels, min_channels),
    )

    min_span = min_channel_skip * (num_corrupted_channels - 1)
    max_span = max_channel_skip * (num_corrupted_channels - 1)
    max_starting_channel = max(num_channels - min_span, 0)
    if max_starting_channel > 0:
        starting_channel_offset = np.random.randint(0, max_starting_channel)
    else:
        starting_channel_offset = 0
    channels = random_list_with_gap_constraints(
        num_corrupted_channels,
        min(max_span, num_channels - starting_channel_offset - 1),
        min_channel_skip,
        max_channel_skip,
    )
    return [
        idx + starting_channel_offset
        for idx in channels
        if (idx + starting_channel_offset) < num_channels
    ]


def create_access_tuple(layout: str, **kwargs) -> Sequence[Any]:
    """
    Returns a tuple that can be used to be accessed ``numpy`` arrays using the axis
    order specified in ``layout``.

    Args
    ----
    * ``layout : str``: A case insensitive string that describes the layout, specifiying the order of the numpy array axes.
                        Each axis is labeled by a character of this string.
    * ``**kwargs``: Each keyword argument contains an index, a slice or a sequence. There must be a 1:1 correspondence between
                    the chars of ``layout`` and the keys of the remaining keyword arguments. The set of all ``kwargs`` must be a valid
                    combination for accessing numpy arrays. For example if there are two or more array kwargs, they must be broadcastable
                    otherwise numpy will throw an error.

    Returns
    ----
    A tuple that can be used for indexing the numpy array. Each axis index is ordered according to the ``layout``, and each
    index is the same as the one provided in the kwargs.
    """
    access = [None] * len(layout)
    kwargs = {k.upper(): v for k, v in kwargs.items()}
    for i, dim in enumerate(layout):
        access[i] = kwargs[dim.upper()]
    return tuple(access)
