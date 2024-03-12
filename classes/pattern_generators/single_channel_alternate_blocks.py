import logging
import math
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from .generator_utils import create_access_tuple

logger = logging.getLogger(__name__)


def single_channel_alternated_block_generator(
    output_shape: Tuple[int], params: Dict[str, Any], layout="CHW"
) -> Optional[List[int]]:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    num_channels = c
    num_values_per_channel = h * w
    mask = np.zeros(output_shape, dtype=np.uint8)

    block_size: int = params["block_size"]
    max_feature_map_size = params["max_feature_map_size"]
    min_block_skip = params["min_block_skip"]
    max_block_skip = params["max_block_skip"]

    random_channel = np.random.randint(0, num_channels)

    num_blocks = int(math.ceil(num_values_per_channel / 2))
    curr_block = 0

    corrupted_positions = []

    while curr_block < num_blocks:
        corrupted_positions += [i + curr_block * block_size for i in range(block_size)]
        block_skip = np.random.randint(min_block_skip, max_block_skip + 1)
        curr_block += block_skip
    h_idxs, w_idxs = np.unravel_index(layout, shape=(h, w))
    access = create_access_tuple(layout, c=random_channel, h=h_idxs, w=w_idxs)
    mask[access] = 1
    return mask
