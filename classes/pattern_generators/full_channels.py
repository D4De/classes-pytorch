from typing import Dict, Any, Sequence, Tuple

import numpy as np

from .generator_utils import (
    create_access_tuple,
    random_channels,
    random_int_from_pct_range,
)


def full_channels_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    num_channels = output_shape[c_dim]
    num_values_per_channel = output_shape[h_dim] * output_shape[w_dim]
    mask = np.zeros(output_shape, dtype=np.uint8)

    channels = random_channels(
        num_channels,
        params["min_channel_skip"],
        params["max_channel_skip"],
        params["max_corrupted_channels"],
        *params["affected_channels_pct"]
    )
    avg_chan_corruption_pct: Tuple[float, float] = params["avg_channel_corruption_pct"]

    for chan in channels:
        num_corr_positions = random_int_from_pct_range(
            num_values_per_channel, *avg_chan_corruption_pct
        )
        positions = np.random.choice(num_corr_positions, num_values_per_channel)
        h_idxs, w_idxs = np.unravel_index(
            positions, (output_shape[h_dim], output_shape[w_dim])
        )
        access = create_access_tuple(layout, c=chan, h=h_idxs, w=w_idxs)
        mask[access] = 1
    return mask
