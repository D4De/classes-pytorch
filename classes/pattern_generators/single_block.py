import logging
from typing import Dict, Any, Sequence, Tuple

import numpy as np

from .generator_utils import create_access_tuple, random_int_from_pct_range

logger = logging.getLogger(__name__)


def single_block_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    mask = np.zeros(output_shape, dtype=np.uint8)

    num_values_per_tensor = c * h * w
    block_corruption_pct: Tuple[float, float] = params["block_corruption_pct"]
    block_size: int = params["block_size"]

    block_start_offset = np.random.randint(
        0, max(1, num_values_per_tensor - block_size)
    )
    cardinality = random_int_from_pct_range(block_size, *block_corruption_pct)
    block_corrupted_idxs = np.random.choice(block_size, cardinality)
    c_idxs, h_idxs, w_idxs = np.unravel_index(
        np.array(
            [
                block_start_offset + idx
                for idx in block_corrupted_idxs
                if block_start_offset + idx < mask.size
            ],
            dtype=np.intp,
        ),
        shape=(c, h, w),
    )

    access = create_access_tuple(layout, c=c_idxs, h=h_idxs, w=w_idxs)

    mask[access] = 1

    return mask
