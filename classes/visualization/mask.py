from math import sqrt
import math
import os
import json
from typing import List, Literal, Tuple, Union


from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

levels = [0, 1, 2]
labels = [""] + ["clean", "corrupted"]
colors = ["white", "red"]
cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="neither")


def split_two(num: int) -> Tuple[int, int]:
    """
    Given a number [num] of plots returns the optimal arrangment for displaying
    the subplot in a 2D grid.
    """
    val = int(math.ceil(sqrt(num)))
    return val, val


def is_square(apositiveint: int) -> bool:
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen:
            return False
        seen.add(x)
    return True

def axs_generator(axs, scene_dim_x):
        # Pick the ax where to draw the channel
    if isinstance(axs, Axes):
        yield axs
        return
    for i in range(axs.size):
        if len(axs.shape) == 1:
            # Multiple plots arranged in a line
            yield axs[i]
        else:
            # Plots arranged in a 2D Grid
            yield axs[i % scene_dim_x, i // scene_dim_x]


def plot_mask(
    mask: np.ndarray,
    layout_type: Literal["CHW", "HWC"],
    output_path: Union[str, None] = None,
    save: bool = False,
    show: bool = True,
    invalidate: bool = False,
    description: str = "",
):
    feat_map_axis = tuple([
        i for i, tensor_ax in enumerate(layout_type) if tensor_ax in ["H", "W"]
    ])
    faulty_channels: List[int] = np.where(mask.any(axis=feat_map_axis))[0].tolist()

    scene_dim_x, scene_dim_y = split_two(len(faulty_channels))

    if len(description) > 0:
        try:
            suptitle = json.dumps(json.loads(description), indent=2)
        except ValueError:
            suptitle = description
    else:
        suptitle = ""
    if not invalidate and output_path and os.path.exists(output_path):
        print('No')
        return

    fig, axs = plt.subplots(scene_dim_x, scene_dim_y, figsize=(6, 7))
    if len(suptitle) > 0:
        plt.suptitle(suptitle, fontsize=8, wrap=True)

    for i, curr_C in enumerate(faulty_channels):
        if layout_type == "CHW":
            slice_diff = mask[curr_C, :, :]
        else:
            slice_diff = mask[:, :, curr_C]

        # Pick the ax where to draw the channel
        if len(faulty_channels) == 1:
            # Single Plot
            curr_axs = axs
        elif len(axs.shape) == 1:
            # Multiple plots arranged in a line
            curr_axs = axs[i]
        else:
            # Plots arranged in a 2D Grid
            curr_axs = axs[i % scene_dim_x, i // scene_dim_x]

        # Show image with diff
        img = curr_axs.imshow(slice_diff, cmap=cmap, norm=norm, interpolation="nearest")

        # Label
        curr_axs.set_title(f"CH {curr_C}", fontsize=9)

    # Remove ticks from all axes
    for ax in axs_generator(axs, scene_dim_x): 
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])



    if show:
        plt.show()

    if save:
        if output_path is None:
            raise ValueError("Output path is required for saving file")

    plt.savefig(output_path)
    plt.close()
