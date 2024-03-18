from typing import Tuple
import numpy as np

from classes.value_generators.float_utils import (
    float_sign,
    get_float_count_from_number,
    uint_repr,
)


def create_fill_generator(fill_value) -> np.ndarray:
    def fill_generator(
        val_range: np.ndarray, size: Tuple[int], dtype=None
    ) -> np.ndarray:
        if dtype is None:
            float_type = val_range.dtype
        else:
            float_type = dtype
        if fill_value == 0.0:
            return np.zeros(size, dtype=float_type)
        data = np.empty(size, dtype=float_type)
        data.fill(fill_value)
        return data

    return fill_generator


def in_range_value_generator(
    val_range: np.ndarray, size: Tuple[int], dtype=None
) -> np.ndarray:
    """
    Generate a numpy array of floats of the same type of ``val_range`` of size ``size``.
    Each float will be inside of the range specified in ``val_range``.

    Args
    ---
    * ``val_range``: A ``numpy.ndarray`` of shape ``(2,)``. The two values of the array contains the range where the number of the output stand.
    * ``size``: The shape of the output array
    * ``dtype``: Optional data type of the output array. If not specifed the dtype of the output will be the same of ``val_range``

    Returns
    ---
    A ``numpy.ndarray`` of shape ``size`` that contains values in the interval ``[val_range[0], val_range[1]]`` (inside the range).
    """
    assert val_range.shape == (
        2,
    ), f"Last dimension of the range tensor must be {size} but instead is {val_range.shape[-1]}"
    a, b = val_range
    if dtype is None:
        float_type = val_range.dtype
    else:
        float_type = dtype
    return np.random.uniform(a, b, size=size, dtype=float_type)


def out_of_range_value_generator(
    val_range: np.ndarray, size: Tuple[int], dtype=None
) -> np.ndarray:
    """
    Generate a numpy array of floats of the same type of ``val_range`` of size ``size``.
    Each float will be out of the range specified in ``val_range``.

    Args
    ---
    * ``val_range``: A ``numpy.ndarray`` of shape ``(2,)``. The two values of the array contains the range of values to be avoided.
    * ``size``: The shape of the output array
    * ``dtype``: Optional data type of the output array. If not specifed the dtype of the output will be the same of ``val_range``

    Returns
    ---
    A ``numpy.ndarray`` of shape ``size`` that contains only floats less than ``val_range[0]`` or greather than``val_range[1]``
    (outside from the range).
    """
    assert val_range.shape == (
        2,
    ), f"Last dimension of the range tensor must be {size} but instead is {val_range.shape[-1]}"
    if dtype is None:
        float_type = val_range.dtype
    else:
        float_type = dtype

    a, b = val_range
    n_bits = val_range.nbytes << 3
    uint_type = np.dtype(f"uint{n_bits}")

    floats_less_than_a, floats_more_than_a, floats_a_to_zero = (
        get_float_count_from_number(a)
    )
    floats_less_than_b, floats_more_than_b, floats_b_to_zero = (
        get_float_count_from_number(b)
    )
    most_positive_float_ieee754 = np.finfo(float_type).max.view(
        uint_type
    )  # for np.float32 ~ +3*10^38
    most_negative_float_ieee754 = np.finfo(float_type).min.view(
        uint_type
    )  # for np.float32 ~ -3*10^38
    positive_zero_ieee754 = np.uint64(0).astype(uint_type)  # for np.float32  0
    negative_zero_ieee754 = np.uint64(1 << (n_bits - 1)).astype(
        uint_type
    )  # for np.float32  2^32
    floats_with_opposite_sign = np.finfo(float_type).max.view(uint_type)  #

    result = np.zeros(size)

    if float_sign(a) * float_sign(b) < 0:
        # a is negative and b is positive
        #  o----------------------------oxxxxxxxxxxxx0xxxxxxxxxxxxo------------------------o
        #  |    (floats_less_than_a)    |  (range to be avoided)  |  (floats_more_than_b)  |
        # -inf                          a            0            b                      +inf
        smaller_than_a = floats_less_than_a
        bigger_than_b = floats_more_than_b
        range_sides_probs = np.array(
            [smaller_than_a, bigger_than_b], type=uint_type
        ) / (smaller_than_a + bigger_than_b)
        small_big_choices = np.random.choice(
            [False, True], size, replace=True, p=range_sides_probs
        )
        bigger_than_b_count = small_big_choices.sum()
        smaller_than_a_count = small_big_choices.size - bigger_than_b_count

        smaller_than_a_values = np.random.randint(
            uint_repr(a),
            most_negative_float_ieee754,
            size=(smaller_than_a_count,),
            dtype=uint_type,
        ).view(float_type)
        bigger_than_b_values = np.random.randint(
            uint_repr(b),
            most_positive_float_ieee754,
            size=(bigger_than_b_count,),
            dtype=uint_type,
        ).view(float_type)

        result[small_big_choices] = smaller_than_a_values
        result[~small_big_choices] = bigger_than_b_values
    elif float_sign(a) < 0:
        # a and b are both negative
        #  o----------------------------oxxxxxxxxxxxxxxxxxxxxxxxxxo----------------------o---------------------------------o
        #  |     (floats_less_than_a)   |  (range to be avoided)  |  (floats_b_to_zero)  |   (floats_with_opposite_sign)   |
        # -inf                          a                         b                      0                                +inf
        smaller_than_a = floats_less_than_a
        bigger_than_b = floats_b_to_zero + floats_with_opposite_sign
        # First choose if each value stands in (-inf,a) or (b,+inf)
        range_sides_probs = np.array(
            [smaller_than_a, bigger_than_b], type=uint_type
        ) / (smaller_than_a + bigger_than_b)
        range_sides_choices = np.random.choice(
            [False, True], size, replace=True, p=range_sides_probs
        )
        # Then choose, for the values between
        sign_probs = (
            np.array([floats_b_to_zero, floats_with_opposite_sign], type=uint_type)
            / bigger_than_b
        )
        sign_choices = np.random.choice([False, True], size, replace=True, p=sign_probs)
        floats_with_opposite_sign_choices = range_sides_choices & sign_choices
        floats_b_to_zero_choices = range_sides_choices & ~sign_choices
        floats_less_than_a_choices = ~range_sides_choices

        floats_with_opposite_sign_values = np.random.randint(
            positive_zero_ieee754,
            most_positive_float_ieee754,
            size=(floats_with_opposite_sign_choices.sum(),),
            dtype=uint_type,
        ).view(float_type)
        floats_b_to_zero_values = np.random.randint(
            uint_repr(b),
            negative_zero_ieee754,
            size=(floats_b_to_zero_choices.sum(),),
            dtype=uint_type,
        ).view(float_type)
        floats_less_than_a_values = np.random.randint(
            most_negative_float_ieee754,
            uint_repr(a),
            size=(floats_less_than_a_choices.sum(),),
            dtype=uint_type,
        ).view(float_type)

        result[floats_with_opposite_sign_choices] = floats_with_opposite_sign_values
        result[floats_b_to_zero_choices] = floats_b_to_zero_values
        result[floats_less_than_a_choices] = floats_less_than_a_values
    else:
        # a and b are both positive
        #  o--------------------------------o-----------------------oxxxxxxxxxxxxxxxxxxxxxxxxo--------------------------o
        #  |  (floats_with_opposite_sign)   |  (floats_a_to_zero)  |  (range to be avoided)  |   (floats_more_than_b)   |
        # -inf                              0                      a                         b                        +inf
        smaller_than_a = floats_with_opposite_sign + floats_a_to_zero
        bigger_than_b = floats_more_than_b
        # First choose if each value stands in (-inf,a) or (b,+inf)
        range_sides_probs = np.array(
            [smaller_than_a, bigger_than_b], type=uint_type
        ) / (smaller_than_a + bigger_than_b)
        range_sides_choices = np.random.choice(
            [False, True], size, replace=True, p=range_sides_probs
        )
        # Then choose, for the values between
        sign_probs = (
            np.array([floats_with_opposite_sign, floats_a_to_zero], type=uint_type)
            / smaller_than_a
        )
        sign_choices = np.random.choice([False, True], size, replace=True, p=sign_probs)
        floats_with_opposite_sign_choices = ~range_sides_choices & ~sign_choices
        floats_a_to_zero_choices = ~range_sides_choices & sign_choices
        floats_more_than_b_choices = range_sides_choices

        floats_with_opposite_sign_values = np.random.randint(
            most_negative_float_ieee754,
            negative_zero_ieee754,
            size=(floats_with_opposite_sign_choices.sum(),),
            dtype=uint_type,
        ).view(float_type)
        floats_a_to_zero_values = np.random.randint(
            negative_zero_ieee754,
            uint_repr(a),
            size=(floats_a_to_zero_choices.sum(),),
            dtype=uint_type,
        ).view(float_type)
        floats_more_than_b_values = np.random.randint(
            uint_repr(b),
            most_positive_float_ieee754,
            size=(floats_more_than_b_choices.sum(),),
            dtype=uint_type,
        ).view(float_type)

        result[floats_with_opposite_sign_choices] = floats_with_opposite_sign_values
        result[floats_a_to_zero_choices] = floats_a_to_zero_values
        result[floats_more_than_b_choices] = floats_more_than_b_values

    return result
