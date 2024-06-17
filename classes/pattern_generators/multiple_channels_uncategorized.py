from typing import Dict, Any, Sequence

import numpy as np
from .generator_utils import (
    clamp,
    create_access_tuple,
    random_channels,
    random_int_from_pct_range,
)


def multiple_channels_uncategorized_generator(
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
        min_channels=2
    )

    for channel in channels:
        num_channel_corrupted_values = random_int_from_pct_range(
            num_values_per_channel, *params["avg_channel_corruption_pct"]
        )
        num_channel_corrupted_values = clamp(
            num_channel_corrupted_values,
            min(params["min_errors_per_channel"], num_values_per_channel),
            min(params["max_errors_per_channel"], num_values_per_channel),
        )
        channel_corr_pos = np.random.choice(
            num_values_per_channel, num_channel_corrupted_values, replace=False
        )
        h_idxs, w_idxs = np.unravel_index(channel_corr_pos, shape=(h, w))
        access = create_access_tuple(layout, c=channel, h=h_idxs, w=w_idxs)
        mask[access] = 1
    return mask
