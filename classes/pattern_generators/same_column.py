import logging
from typing import Dict, Any, Sequence

import numpy as np

from .generator_utils import create_access_tuple, random_channels

logger = logging.getLogger(__name__)


def same_row_generator(
    output_shape: Sequence[int], params: Dict[str, Any], layout="CHW"
) -> np.ndarray:
    c_dim, h_dim, w_dim = layout.index("C"), layout.index("H"), layout.index("W")
    c, h, w = output_shape[c_dim], output_shape[h_dim], output_shape[w_dim]
    mask = np.zeros(output_shape, dtype=np.uint8)

    num_channels = c
    col_corruption_pct = params["col_corruption_pct"]
    max_cardinality = params["max_cardinality"]
    min_value_skip = params["min_value_skip"]
    max_value_skip = params["max_value_skip"]
    num_rows = h
    num_cols = w

    rows_indexes = random_channels(
        num_rows, min_value_skip, max_value_skip, max_cardinality, *col_corruption_pct
    )

    random_channel = np.random.randint(0, num_channels)
    random_col = np.random.randint(0, num_cols)

    corrupted_positions = [idx * num_cols + random_col for idx in rows_indexes]
    h_idxs, w_idxs = np.ravel_multi_index(corrupted_positions, (h, w))
    access = create_access_tuple(layout, c=random_channel, h=h_idxs, w=w_idxs)
    mask[access] = 1

    return mask
