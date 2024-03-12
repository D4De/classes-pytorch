from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def single_generator(
    output_shape: Tuple[int], params: Dict[str, Any], layout="CHW"
) -> Optional[List[int]]:
    mask = np.zeros(output_shape)
    value = np.random.randint(0, mask.size)
    mask[np.unravel_index(value, mask.shape)] = 1
    return mask
