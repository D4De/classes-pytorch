import math
from typing import Any, Dict, Sequence
import numpy as np

from .generator_utils import create_access_tuple, random_channels


def skip_4_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    num_channels = c
    num_values_per_channel = h * w
    mask = np.zeros(output_shape, dtype=np.uint8)

    # 1. Select which channels to affect
    corr_channels = random_channels(
        num_channels,
        params["min_channel_skip"],
        params["max_channel_skip"],
        params["max_corrupted_channels"],
        *params["affected_channels_pct"]
    )

    # 2. Select potential positions that can be affected by corruption,
    # These positions are spaced regulary every ``skip_amount`` and will be the same for each channel
    skip_amount = params[
        "skip_amount"
    ]  # How many location there is a corrupted location
    unique_map_indexes = params[
        "unique_channel_indexes"
    ]  # Number of unique locations that are corrupted stacking all channels

    remainder = np.random.randint(
        0, skip_amount
    )  # if skip_amount = 4 the index of the first location modulo 4 can be 0,1,2,3
    # Pick from which position (expressed as a channel-raveled index) start the "draw"
    max_starting_map_offset = 1 + max(
        int(
            math.floor(
                (num_values_per_channel - remainder - skip_amount * unique_map_indexes)
                / skip_amount
            )
        ),
        0,
    )
    starting_map_offset = np.random.randint(0, max_starting_map_offset)
    # List of potential corrupted indexes in each one of the corrupted channels selected at step 1
    candidate_locations_per_channel = np.arange(
        starting_map_offset * skip_amount,
        (unique_map_indexes + starting_map_offset) * skip_amount,
        skip_amount,
    )
    h_coords, w_coords = np.unravel_index(candidate_locations_per_channel, (h, w))

    # 3. Select which positions are effectively corrupted
    pct_corrupted = np.random.uniform(*params["indexes_corruption_pct"])
    for chan in corr_channels:
        access = create_access_tuple(layout, c=chan, h=h_coords, w=w_coords)
        mask[access] = (
            np.random.uniform(
                size=(
                    len(
                        candidate_locations_per_channel,
                    )
                )
            )
            < pct_corrupted
        )

    return mask
