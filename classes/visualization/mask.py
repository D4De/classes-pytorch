from math import sqrt
import math
import os
import json
from typing import List, Literal, Optional, Sequence, Tuple, Union


from matplotlib.axes import Axes
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np



def split_two(num: int) -> Tuple[int, int]:
    """
    Given a number [num] of plots returns the optimal arrangment for displaying
    the subplot in a square 2D grid.
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


def axs_generator(axs : Union[Axes, np.ndarray], scene_dim_x : int):
    """

    Args
    ---
    * ``axs : Axes | np.ndarray[Axes]``. The axes object or array returned by matplotlib subplots function
    * ``scene_dim_x : int``. Number of columns of plots in the subplot.

    Returns
    ---
    A generator that allows to iterate all the axes independelty from
    the fact that ``axs`` is a single Axes, or a 1d array of Axes or a 2d array of Axes.
    """
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
    output_path: Optional[str] = None,
    save: bool = False,
    show: bool = True,
    invalidate: bool = False,
    description: str = "",
    labels : Sequence[str] = ["clean", "corrupted"],
    colors : Sequence[str] = ["white", "red"]
):
    """
    Plots the corrupted values 3d masks using matplotlib. 
    The 3d array is decomposed in various 2d heatmap plots where each pixel is colored depending on its class (for example clean/corrupted, or its ValueClass).
    The plots can be shown in a window a saved to an image.

    Args
    ---
    * ``mask : np.ndarray``. A 3d ndarray of integer, each representing the class of the pixel. The classes vary depending on the context for which this function is used.
    * ``layout_type : "CHW" | "HWC"``. Specify if the tensor is saved with the torch convention (channel first - CHW) or the tensorflow convention (channel last - HWC). Default is CHW.
    * ``output_path : Optional[str]``. Specify the path where to save the resulting image. If save is ``True`` it must be specified.
    * ``save : bool``. If ``True`` the image of the generated plots will be saved to ``output_path``
    * ``show : bool``. If ``True`` the plot will be shown in a matplotlib window.
    * ``invalidate``. If ``False`` the plot will not be generated if already exist another plot. Defaults to ``False``.
    * ``description``. The description that will be put as the global title of the image.
    * ``labels : Sequence[str]``. A list of label names for the legend
    * ``colors : Sequence``. A list of colors for the various classes.

    NOTE: labels and colors must a length equal to the number of classes in the ``mask``. If there are N classes, mask must contain int values from 0 to N-1.
    """

    if len(labels) != len(colors):
        raise ValueError(f'labels and colors must have the same len. len(labels)={len(labels)} len(colors)={len(colors)}')
    labels = [""] + list(labels)
    levels = list(range(len(labels)))

    cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors, extend="neither")


    feat_map_axis = tuple(
        [i for i, tensor_ax in enumerate(layout_type) if tensor_ax in ["H", "W"]]
    )
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
        print("Plots already exist.")
        return

    fig, axs = plt.subplots(max(1, scene_dim_x), max(1, scene_dim_y), figsize=(6, 7))
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
