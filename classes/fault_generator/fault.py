from dataclasses import dataclass, astuple
from typing import Any, Dict, Sequence

import numpy as np

@dataclass
class Fault:
    corrupted_value_mask : np.ndarray
    corrupted_values : np.ndarray
    spatial_pattern_name : str
    sp_parameters : Dict[str, Any]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))

@dataclass
class FaultBatch:
    corrupted_value_mask : np.ndarray
    corrupted_values : np.ndarray
    corrupted_values_index : np.ndarray
    spatial_pattern_names : Sequence[str]
    sp_parameters : Sequence[Dict[str, Any]]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))
    
    def get_element(self, idx : int):
        value_begin, value_end = self.corrupted_values_index[idx:idx+1]

        return Fault(
            self.corrupted_value_mask[idx],
            self.corrupted_values[value_begin:value_end],
            self.spatial_pattern_names[idx],
            self.sp_parameters[idx]
        )
    