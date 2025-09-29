import logging
import math
from typing import Dict, Any, Sequence

import numpy as np

from .generator_utils import create_access_tuple

logger = logging.getLogger(__name__)


def single_channel_alternated_block_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
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

    num_blocks = int(math.ceil(num_values_per_channel / block_size))
    curr_block = 0

    corrupted_blocks = (
        []
    )  # Contains the id of the blocks that will be entirely corrupted

    while curr_block < num_blocks:
        corrupted_blocks.append(curr_block)

        block_skip = np.random.randint(
            min_block_skip, max_block_skip + 1
        )  # Random skip depending on parameters
        curr_block += block_skip

    # Calculate where the blocks start (with flattened channel index)
    corrupted_blocks_start = (
        np.expand_dims(np.array(corrupted_blocks), axis=1) * block_size
    )  # shape (n_blocks, 1)

    positions = corrupted_blocks_start + np.arange(
        0, block_size
    )  # shape (n_blocks, block_size)
    corrupted_positions = positions.flatten()  # shape (n_blocks * block_size)
    corrupted_positions[corrupted_positions >= num_values_per_channel] = (
        corrupted_positions[0]
    )

    h_idxs, w_idxs = np.unravel_index(corrupted_positions, shape=(h, w))
    access = create_access_tuple(layout, c=random_channel, h=h_idxs, w=w_idxs)
    mask[access] = 1
    return mask
