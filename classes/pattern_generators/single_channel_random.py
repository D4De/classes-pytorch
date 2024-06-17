import logging
from typing import Dict, Any, Sequence

import numpy as np

from .generator_utils import create_access_tuple, random_channels

logger = logging.getLogger(__name__)


def single_channel_random_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:

    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    num_channels = c
    num_values_per_channel = h * w
    mask = np.zeros(output_shape, dtype=np.uint8)

    channel_corruption_pct = params["channel_corruption_pct"]
    max_cardinality = params["max_cardinality"]
    min_value_skip = params["min_value_skip"]
    max_value_skip = params["max_value_skip"]

    chan_positions = random_channels(
        num_values_per_channel,
        min_value_skip,
        max_value_skip,
        max_cardinality,
        *channel_corruption_pct
    )
    h_idxs, w_idxs = np.unravel_index(chan_positions, (h, w))
    random_channel = np.random.randint(0, num_channels)
    access = create_access_tuple(layout, c=random_channel, h=h_idxs, w=w_idxs)
    mask[access] = 1
    return mask
