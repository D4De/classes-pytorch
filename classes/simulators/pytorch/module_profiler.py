from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils.data import DataLoader

from collections import defaultdict
from operator import itemgetter


import numpy as np

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def module_shape_profiler(
    module: nn.Module,
    input_data: Optional[torch.Tensor] = None,
    input_shape: Optional[Sequence[int]] = None,
    device=DEFAULT_DEVICE,
    module_filter_fn: Callable[[str, nn.Module], bool] = lambda module, name: True
) -> Dict[str, List[int]]:
    """
    Executes a forward pass in a Module to determine the shape of all
    the children modules at every nesting level.

    The function takes in input the module and alternatively one
    of `input_data` and `input_shape`. And returns a dictionary containing all the shapes of submodules.

    Args
    ----
    * `module : nn.Module`. The module to be profiled
    * `input_data : Tensor | None`. A dummy input accepted from the module
    * `input_shape : Tensor | None`. The shape of an input tensor accepted by the network. This must be specified only if `input_data` is not specified.
    * `device`: The torch device where the test inference is executed. If not specified defaults to cuda if available, otherwise cpu. Note that the model
                will be moved to device as a side-effect of this function.
    * `module_filter_fn : Callable[[str, nn.Module], bool]`. A function that takes in input the module name and the module itself and returns a boolean that says
                whether the profiling should happen in that layer. If not specified, the output shape of all modules will be profiled.
                
    Returns
    ---
    A dictionary that has the submodules fully qualified names as keys and their corresponding output shapes
    as the corresponding values.

    Raises
    ---
    * `ValueError` if both or none of `input_shape` and `input_data` are specified.
    """

    shape_index = {}

    # This hook will be added at each submodule and:
    # * gets the size of the output after the execution of a submodule
    # * puts in the result dictionary (shape_index)
    # * does not modify the output (returning it as given)
    def _make_shape_profile_hook(name):
        def _shape_profile_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                shape_index[name] = output.size()
            # Do not modify the output
            return output

        return _shape_profile_hook


    if input_data and not input_shape:
        input_shape = input_data.shape
        input_data = input_data.to(device)
    elif not input_data and input_shape:
        input_data = torch.normal(0.0, 1.0, input_shape).to(device)
    else:
        raise ValueError("One and only one between input_data and input_shape must be specified.")
    
    module.to(device)

    module.eval()
    # Store the handles to remove the hook after the profiling
    hook_handles: List[RemovableHandle] = []
    try:
        for name, mod in module.named_modules():
            if module_filter_fn(name, mod):
                handle = mod.register_forward_hook(_make_shape_profile_hook(name))
                hook_handles.append(handle)
        with torch.no_grad():
            module(input_data)
    finally:
        # Restore the network as it was before (removing hooks applied in this function)
        for handle in hook_handles:
            handle.remove()

    return shape_index


def module_range_profiler(
    network: nn.Module,
    dataloader: DataLoader,
    network_input_fn: Callable = itemgetter(0),
    torch_dtype=torch.float32,
    np_output_dtype=np.float32,
    device=DEFAULT_DEVICE,
) -> Dict[str, np.ndarray]:

    min_value_per_module = defaultdict(
        lambda: torch.tensor(np.infty, dtype=torch_dtype).to(device)
    )
    max_value_per_module = defaultdict(
        lambda: torch.tensor(-np.infty, dtype=torch_dtype).to(device)
    )

    def _make_range_profile_hook(module_name):
        def _range_profile_hook(module, input, output):
            min_value_per_module[module_name] = torch.min(
                min_value_per_module[module_name], torch.min(output)
            )
            max_value_per_module[module_name] = torch.max(
                max_value_per_module[module_name], torch.max(output)
            )
            # Do not modify the output
            return output

        return _range_profile_hook

    network.to(device)

    hook_handles: List[RemovableHandle] = []

    try:
        for name, module in network.named_modules():
            handle = module.register_forward_hook(_make_range_profile_hook(name))
            hook_handles.append(handle)
        with torch.no_grad():
            for data in dataloader:
                network_input = network_input_fn(data)
                network_input = network_input.to(device)
                output = network(network_input)

    finally:
        for handle in hook_handles:
            handle.remove()

    result = {}

    for module_name in min_value_per_module.keys():
        min_val = min_value_per_module[name].cpu().numpy()
        max_val = max_value_per_module[name].cpu().numpy()

        result[module_name] = np.array([min_val, max_val], dtype=np_output_dtype)

    return result
