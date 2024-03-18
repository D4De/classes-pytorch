from typing import Tuple
import numpy as np


def uint_repr(x: np.floating) -> np.integer:
    n_bits = x.nbytes << 3
    uint_type = np.dtype(f"uint{n_bits}")
    return x.view(uint_type)


def float_sign(x: np.floating) -> np.int8:
    assert isinstance(
        x, np.floating
    ), f"input must be a numpy floating but is {type(x)}"
    n_bits = x.nbytes << 3
    target_type = np.dtype(f"uint{n_bits}")
    negative_zero_ieee754 = np.uint64(1 << (n_bits - 1)).astype(target_type)
    x_as_uint = x.view(np.uint32)
    return np.int8(+1 if x_as_uint < negative_zero_ieee754 else -1)


def get_float_count_from_number(x: np.floating) -> Tuple[int, int, int]:
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
