from typing import Tuple
import numpy as np


def uint_repr(x: np.floating) -> np.integer:
    """
    Calculates the bit representation of a numpy float, viewed as an integer.

    NOTE: This function may not work for floating point with more than 64 bits (doubles) because the corresponding
    uint types are not implementented or viceversa.

    Args
    ---
    * ``x : np.floating``. The float to view as bit representation

    Returns
    ---
    A numpy unsigned integer with the same number of bits of the float ``x``.
    The value returned is the uint that has the same binary representation of the floating point number.

    """
    n_bits = x.nbytes << 3
    uint_type = np.dtype(f"uint{n_bits}")
    return x.view(uint_type)


def float_sign(x: np.floating) -> np.int8:
    """
    Extracts the sign bit of a floating point number.
    +
    Args
    ---
    * ``x : np.floating``. The float whose sign will be extracted.

    Returns
    ---
    A ``np.int8`` ``+1`` if the float is positive or else ``-1`` if the float is negative.
    """
    assert isinstance(
        x, np.floating
    ), f"input must be a numpy floating but is {type(x)}"

    return np.int8(+1 if not np.signbit(x) else -1)

 
def get_float_count_from_number(x: np.floating) -> Tuple[int, int, int]:
    """
    Returns the number of floating point numbers that are respectively
    smaller than ``x``, bigger than ``x`` and the number of floating
    points from zero to ``x``
    NOTE: This function does not work for float with more than 64 bits.

    Args
    ---
    * ``x : np.floating``. A float

    Returns
    ---
    A tuple containing three python ints.
    * The number of floating point numbers smaller than x
    * The number of floating point numbers bigger than x
    * The number of floating point numbers that separate 0 to x
    """
    assert isinstance(
        x, np.floating
    ), f"input must be a numpy floating but is {type(x)}"
    n_bits = x.nbytes << 3
    uint_type = np.dtype(f"uint{n_bits}")
    floats_with_opposite_sign = np.finfo(x).max.view(uint_type)
    positive_zero_ieee754 = np.uint64(0).astype(uint_type)
    negative_zero_ieee754 = np.uint64(1 << (n_bits - 1)).astype(uint_type)
    most_positive_float_ieee754 = np.finfo(x).max.view(uint_type)
    most_negative_float_ieee754 = np.finfo(x).min.view(uint_type)

    x_as_uint = np.array(x, dtype=np.float32).view(np.uint32)
    if float_sign(x) >= 0:
        floats_from_pos_zero = x_as_uint - positive_zero_ieee754
        floats_to_max_float = most_positive_float_ieee754 - x_as_uint
        floats_less_than_x = floats_with_opposite_sign + floats_from_pos_zero
        floats_more_than_x = floats_to_max_float
        return floats_less_than_x, floats_more_than_x, floats_from_pos_zero
    else:
        floats_from_neg_zero = x_as_uint - negative_zero_ieee754
        # print(f'{floats_from_neg_zero=}')
        floats_to_min_float = most_negative_float_ieee754 - x_as_uint
        # print(f'{floats_to_min_float=}')
        floats_less_than_x = floats_to_min_float
        # print(f'{floats_less_than_x=}')
        floats_more_than_x = floats_from_neg_zero + floats_with_opposite_sign
        # print(f'{floats_more_than_x=}')
        return floats_less_than_x, floats_more_than_x, floats_from_neg_zero
