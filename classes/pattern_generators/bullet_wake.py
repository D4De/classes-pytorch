from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from .generator_utils import create_access_tuple, random_channels


def bullet_wake_generator(
    output_shape: Tuple[int], params: Dict[str, Any], layout="CHW"
) -> Optional[List[int]]:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    num_channels = c
    num_values_per_channel = h * w
    mask = np.zeros(output_shape, dtype=np.uint8)
    channels = random_channels(
        num_channels,
        params["min_channel_skip"],
        params["max_channel_skip"],
        params["max_corrupted_channels"],
        *params["affected_channels_pct"],
        min_channels=2
    )
    random_position = np.random.randint(0, num_values_per_channel)
    x, y = np.unravel_index(random_position, (h, w))
    access = create_access_tuple(layout, c=channels, h=x, w=y)
    mask[access] = 1
    return mask
