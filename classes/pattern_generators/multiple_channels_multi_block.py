import logging
from typing import Dict, Any, Sequence, Tuple

import numpy as np
import math

from .generator_utils import (
    create_access_tuple,
    random_channels,
    random_int_from_pct_range,
)

logger = logging.getLogger(__name__)


def multiple_channels_multi_block_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    mask = np.zeros(output_shape, dtype=np.uint8)

    num_channels = c
    num_values_per_channel = h * w
    channels = random_channels(
        num_channels,
        params["min_channel_skip"],
        params["max_channel_skip"],
        params["max_corrupted_channels"],
        *params["affected_channels_pct"],
        min_channels=2,
    )
    avg_block_corruption_pct: Tuple[float, float] = params["avg_block_corruption_pct"]
    block_size: int = params["block_size"]

    num_blocks_per_channel = num_values_per_channel // block_size
    # Consider the remainder block valid only if is at least half of the normal block length
    if (
        num_values_per_channel % block_size >= block_size // 2
        or num_blocks_per_channel == 0
    ):
        num_blocks_per_channel += 1
        remainder_block_included = True
    else:
        remainder_block_included = False

    random_block = np.random.randint(0, num_blocks_per_channel)
    # picked_block_suze contains the real block size of the selected block
    # It is equal block_size unless the block is a remainder
    if random_block == num_blocks_per_channel - 1 and remainder_block_included:
        picked_block_size = num_values_per_channel % block_size
    else:
        picked_block_size = block_size

    for chan in channels:
        num_corr_positions = max(
            block_size // 2,
            random_int_from_pct_range(picked_block_size, *avg_block_corruption_pct),
        )
        positions = np.random.choice(picked_block_size, num_corr_positions)
        chan_corr_pos = positions + random_block * block_size
        h_idxs, w_idxs = np.unravel_index(chan_corr_pos, (h, w))
        access = create_access_tuple(layout, c=chan, h=h_idxs, w=w_idxs)
        mask[access] = 1
    return mask
