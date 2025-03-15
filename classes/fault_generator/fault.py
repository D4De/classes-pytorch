from dataclasses import dataclass, astuple
from typing import Any, Dict, Sequence

import numpy as np


@dataclass
class Fault:
    """
    Characterizes a single Fault, independetly from the front-end deep learning framework used for the simulations.
    """

    corrupted_value_mask: np.ndarray
    """
    A 3d numpy array of uint8 with the shape of the intermediate tensor. The values stored in the
    tensor represent the ids of the `ValueClasses`, if 0 the position is not corrupted.
    """
    
    corrupted_values: np.ndarray
    """
    A 1d array of `float32` (or the type of the intermediate tensors). Contains the actual values that
    will replace the golden values. The length of the array is equal to the number of nonzero values `corrupted_value_mask`.
    The position of the values is specified `corrupted_value_mask`.
    """

    spatial_pattern_name: str
    """
    The name of the spatial pattern.
    """

    sp_parameters: Dict[str, Any]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))


@dataclass
class FaultBatch:
    """
    Characterizes a batch of multiple Faults, independetly from the front-end deep learning framework used for the simulations.
    """

    corrupted_value_mask: np.ndarray
    corrupted_values: np.ndarray
    corrupted_values_index: np.ndarray
    spatial_pattern_names: Sequence[str]
    sp_parameters: Sequence[Dict[str, Any]]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))
    
    def __len__(self):
        return len(self.spatial_pattern_names)

    def get_element(self, idx: int):
        value_begin, value_end = self.corrupted_values_index[idx : idx + 1]

        return Fault(
            self.corrupted_value_mask[idx],
            self.corrupted_values[value_begin:value_end],
            self.spatial_pattern_names[idx],
            self.sp_parameters[idx],
        )
