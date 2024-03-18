from typing import Dict, Any, Sequence

import numpy as np
from .generator_utils import (
    clamp,
    create_access_tuple,
    random_channels,
    random_int_from_pct_range,
)


def shattered_channel_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    mask = np.zeros(output_shape, dtype=np.uint8)
    num_channels = c
    num_values_per_channel = h * w
    corr_channels = random_channels(
        num_channels,
        params["min_channel_skip"],
        params["max_channel_skip"],
        params["max_corrupted_channels"],
        *params["affected_channels_pct"],
        min_channels=2
    )

    common_position = np.random.randint(0, num_values_per_channel)

    for channel in corr_channels:
        span_width = clamp(
            np.random.randint(params["min_span_width"], params["max_span_width"] + 1),
            1,
            num_values_per_channel,
        )
        span_begin = np.random.randint(
            max(0, common_position - span_width),
            max(
                min(num_values_per_channel - 1, num_values_per_channel - span_width), 1
            ),
        )
        channel_num_corr_pos = random_int_from_pct_range(
            span_width, *params["avg_span_corruption_pct"]
        )
        channel_corr_pos = (
            np.random.choice(span_width, channel_num_corr_pos, replace=False)
            + span_begin
        )
        h_idxs, w_idxs = np.ravel_multi_index(channel_corr_pos, dims=(h, w))
        h_comm, w_comm = np.ravel_multi_index(common_position, dims=(h, w))
        access = create_access_tuple(layout, c=channel, h=h_idxs, w=w_idxs)
        comm_access = create_access_tuple(layout, c=channel, h=h_comm, w=w_comm)
        mask[access] = 1
        mask[comm_access] = 1

    return mask
