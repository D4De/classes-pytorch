from contextlib import contextmanager
from typing import Callable, Dict, Mapping
import numpy as np
import torch
import torch.nn as nn

from torch.utils.hooks import RemovableHandle


ACTIVATIONS = [
    cls
    for name, cls in nn.modules.activation.__dict__.items()
    if isinstance(cls, type) and issubclass(cls, nn.Module)
]
ACTIVATIONS.remove(nn.Module)
OTHER_RANGER_MODULES = [nn.MaxPool2d, nn.AvgPool2d]

DEFAULT_RANGER_MODULES = tuple(ACTIVATIONS + OTHER_RANGER_MODULES)


def get_ranger_default_protected_module_types():
    return tuple(DEFAULT_RANGER_MODULES)


def create_ranger_hook(layer_range_min: float, layer_range_max: float):
    def _ranger_hook(module, input, output):
        output[(output < layer_range_min) | (output > layer_range_max)] = 0.0
        return output

    return _ranger_hook


def create_clipper_hook(layer_range_min: float, layer_range_max: float):
    def _clipper_hook(module, input, output):
        return torch.clip(output, min=layer_range_min, max=layer_range_max)

    return _clipper_hook


def apply_ranger(
    module: nn.Module,
    modules_range_profile: Mapping[str, np.ndarray],
    hook_factory: Callable = create_ranger_hook,
    ranger_modules_types: tuple[type[nn.Module], ...] = DEFAULT_RANGER_MODULES,
) -> Mapping[str, RemovableHandle]:
    """
    Applies Ranger hook to all modules of certain types.

    Args
    ---
    * ``module``. The super module where to apply the ranger. The submodules will be iterated using ``module.named_modules()`` torch method.
    * ``modules_ranger_profile``. A map between fully qualified sublayer name (relative to ``module`` arg), and a 1d nparray of size 2 containing
        the operating range of the module. 
    * ``hook_factory``. A method that takes in input two floats represetin respectively the minimum and the maximum bounds of the operating
        range and returns a pytorch forward hook function that applies ranger with the bounds passed to the function. 
        Some default hook factories are available in classes.simulators.pytorch.ranger.ranger_hook:
        * create_ranger_hook (default): Zeroes the values out of range
        * create_clipper_hook: Clips the value out of range to the min or to the max of the operating range.
    * ``ranger_modules_types``. An array of types. Ranger will be applied to every module that is an instance of at least one type of this list.
        TODO: Replace this with a filter that takes the qualifed name and the module object (from module.named_modules()) and returns True
        if ranger hook should be applied after the layer.

    Returns
    ---
    A Map between fully qualified names of layers with ranger applied and their hook RemovableHandles. The handles must be used to 
    remove the hooks from the network when ranger is not needed anymore.

    Raises
    ---
    * ``KeyError`` if a layer that qualifies for ranger is missing its operating range from modules_range_profile map.
    """
    ranger_handles_dict = {}
    for name, module in module.named_modules():
        if isinstance(module, ranger_modules_types):
            range_min, range_max = modules_range_profile[name]
            ranger_hook = hook_factory(range_min, range_max)
            ranger_handle = module.register_forward_hook(ranger_hook)
            ranger_handles_dict[name] = ranger_handle

    return ranger_handles_dict


@contextmanager
def applied_ranger_hooks(
    module: nn.Module,
    modules_range_profile: Mapping[str, np.ndarray],
    hook_factory: Callable = create_ranger_hook,
    ranger_modules=get_ranger_default_protected_module_types(),
):
    """
    Applies to a pytorch module inside a context manager to be used in a ```with ... as ...:`` block.
    Inside the block ranger is active and applied to the pytorch module. When exiting the context for any reason
    the ranger hooks are removed and the network returns as before the application of ranger

    Args
    ---
    * ``module``. The super module where to apply the ranger. The submodules will be iterated using ``module.named_modules()`` torch method.
    * ``modules_ranger_profile``. A map between fully qualified sublayer name (relative to ``module`` arg), and a 1d nparray of size 2 containing
        the operating range of the module. 
    * ``hook_factory``. A method that takes in input two floats represetin respectively the minimum and the maximum bounds of the operating
        range and returns a pytorch forward hook function that applies ranger with the bounds passed to the function. 
        Some default hook factories are available in classes.simulators.pytorch.ranger.ranger_hook:
        * create_ranger_hook (default): Zeroes the values out of range
        * create_clipper_hook: Clips the value out of range to the min or to the max of the operating range.
    * ``ranger_modules_types``. An array of types. Ranger will be applied to every module that is an instance of at least one type of this list.
        TODO: Replace this with a filter that takes the qualifed name and the module object (from module.named_modules()) and returns True
        if ranger hook should be applied after the layer.

    Returns
    ---
    A Map between fully qualified names of layers with ranger applied and their hook RemovableHandles. The handles must be used to 
    remove the hooks from the network when ranger is not needed anymore.

    Raises
    ---
    * ``KeyError`` if a layer that qualifies for ranger is missing its operating range from modules_range_profile map.
    """
    ranger_handles = apply_ranger(
        module, modules_range_profile, hook_factory, ranger_modules
    )
    try:
        yield list(ranger_handles.keys())
    finally:
        for handle in ranger_handles.values():
            handle.remove()
