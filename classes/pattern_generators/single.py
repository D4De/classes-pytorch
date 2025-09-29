from typing import Any, Dict, Sequence
import numpy as np


def single_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    mask = np.zeros(output_shape)
    value = np.random.randint(0, mask.size)
    mask[np.unravel_index(value, mask.shape)] = 1
    return mask
