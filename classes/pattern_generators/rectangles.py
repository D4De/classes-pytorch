from typing import Dict, Any, Sequence

import numpy as np

from .generator_utils import create_access_tuple, random_channels


def rectangles_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    mask = np.zeros(output_shape, dtype=np.uint8)
    num_channels = c
    channels = random_channels(
        num_channels,
        params["min_channel_skip"],
        params["max_channel_skip"],
        params["max_corrupted_channels"],
        *params["affected_channels_pct"]
    )

    channel_height = h
    channel_width = w
    rectangle_width = min(params["rectangle_width"], channel_width)
    rectangle_height = min(params["rectangle_height"], channel_height)

    random_top = np.random.randint(0, max(channel_height - rectangle_height, 1))
    random_left = np.random.randint(0, max(channel_width - rectangle_width, 1))

    top_left_position = random_top * channel_width + random_left

    for chan in channels:
        rectangle_positions = [
            top_left_position + h * rectangle_width + w
            for h in range(rectangle_height)
            for w in range(rectangle_width)
        ]

        h_idxs, w_idxs = np.unravel_index(rectangle_positions, (h, w))
        access = create_access_tuple(layout, c=chan, h=h_idxs, w=w_idxs)
        mask[access] = 1

    return mask
