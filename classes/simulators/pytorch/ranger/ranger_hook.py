from contextlib import contextmanager
from typing import Callable, Dict, Mapping
import numpy as np
import torch
import torch.nn as nn

from torch.utils.hooks import RemovableHandle


ACTIVATIONS = [cls for name, cls in nn.modules.activation.__dict__.items() if isinstance(cls, type) and issubclass(cls, nn.Module)]
ACTIVATIONS.remove(nn.Module)
OTHER_RANGER_MODULES = [nn.MaxPool2d, nn.AvgPool2d]

DEFAULT_RANGER_MODULES = tuple(ACTIVATIONS + OTHER_RANGER_MODULES)


def get_ranger_default_protected_module_types():
    return tuple(DEFAULT_RANGER_MODULES)

def create_ranger_hook(layer_range_min : float, layer_range_max : float):
    def _ranger_hook(module, input, output):
        output[(output < layer_range_min) | (output > layer_range_max)] = 0.0
        return output
    return _ranger_hook

def create_clipper_hook(layer_range_min : float, layer_range_max : float):
    def _clipper_hook(module, input, output):
        return torch.clip(output, min=layer_range_min, max=layer_range_max)
    return _clipper_hook

def apply_ranger(
        module : nn.Module, 
        modules_range_profile : Mapping[str, np.ndarray], 
        hook_factory : Callable = create_ranger_hook,  
        ranger_modules_types : tuple[type[nn.Module], ...] = DEFAULT_RANGER_MODULES
    ) -> Mapping[str, RemovableHandle]:
    """
    Apply ranger hook 
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
    module : nn.Module, 
    modules_range_profile : Mapping[str, np.ndarray], 
    hook_factory : Callable = create_ranger_hook,  
    ranger_modules = get_ranger_default_protected_module_types()
):
    ranger_handles = apply_ranger(module, modules_range_profile, hook_factory, ranger_modules)
    try:
        yield list(ranger_handles.keys())
    finally:
        for handle in ranger_handles.values():
            handle.remove()