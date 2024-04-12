from typing import Any, Callable, Dict, Mapping, Sequence
from .bullet_wake import bullet_wake_generator
from .multiple_channels_multi_block import multiple_channels_multi_block_generator
from .same_column import same_row_generator
from .shattered_channel import shattered_channel_generator
from .single_channel_alternate_blocks import single_channel_alternated_block_generator
from .single import single_generator
from .full_channels import full_channels_generator
from .rectangles import rectangles_generator
from .same_row import same_row_generator
from .single_block import single_block_generator
from .single_channel_random import single_channel_random_generator
from .skip_4 import skip_4_generator
from .multiple_channels_uncategorized import multiple_channels_uncategorized_generator

import numpy as np

PatternGenerator = Callable[[Sequence[int], Dict[str, Any], str], np.ndarray]


def get_default_generators() -> Mapping[str, PatternGenerator]:
    return {
        "bullet_wake": bullet_wake_generator,
        "multi_channel_block": multiple_channels_multi_block_generator,
        "same_column": same_row_generator,
        "shattered_channel": shattered_channel_generator,
        "quasi_shattered_channel": shattered_channel_generator,
        "single_channel_alternated_blocks": single_channel_alternated_block_generator,
        "multiple_channels_uncategorized": multiple_channels_uncategorized_generator,
        "single": single_generator,
        "full_channels": full_channels_generator,
        "rectangles": rectangles_generator,
        "same_row": same_row_generator,
        "single_block": single_block_generator,
        "single_channel_random": single_channel_random_generator,
        "skip_4": skip_4_generator,
    }
