from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from torch.utils.data import DataLoader

from collections import defaultdict
from operator import itemgetter


import numpy as np

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def network_shape_profiler(
    network: nn.Module,
    input_data: Optional[torch.Tensor] = None,
    input_shape: Optional[Sequence[int]] = None,
    device=DEFAULT_DEVICE,
) -> Dict[str, List[int]]:

    shape_index = {}

    def _make_shape_profile_hook(name):
        def _shape_profile_hook(module, input, output):
            shape_index[name] = list(output.shape)
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
    
    network.to(device)

    hook_handles: List[RemovableHandle] = []

    try:
        for name, module in network.named_modules():
            handle = module.register_forward_hook(_make_shape_profile_hook(name))
            hook_handles.append(handle)
        network(input_data)
    finally:
        for handle in hook_handles:
            handle.remove()

    return shape_index


def network_range_profiler(
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
