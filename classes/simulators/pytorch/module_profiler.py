from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler
from torch.utils.hooks import RemovableHandle
from torch.utils.data import DataLoader

from collections import defaultdict
from operator import itemgetter

from tqdm import tqdm

import numpy as np
import json
import csv
import os
import gc

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def module_shape_profiler(
    module: nn.Module,
    input_data: Optional[torch.Tensor] = None,
    input_shape: Optional[Sequence[int]] = None,
    device=DEFAULT_DEVICE,
    module_filter_fn: Callable[[str, nn.Module], bool] = lambda module, name: True,
    profile_input_shapes = False
) -> Mapping[str, List[int]]:
    """
    Executes a forward pass in a Module to determine the input or output shapes of all
    the children modules at every nesting level.

    The function takes in input the module and alternatively one
    of `input_data` and `input_shape`. And returns a dictionary containing all the shapes of submodules.

    VERY IMPORTANT NOTE: Each layer in the module MUST not be reused multiple time.
    EACH operator defined in init MUST BE APPLIED ONLY ONCE IN THE WHOLE NETWORK

    Also do not use functional operators if you want to inject them. Define always operators in the __init__

    WHY THESE RESTRICTIONS?
    Before generating faults, CLASSES profiles modules output shapes, and uses their fully qualified
    name to index them in the result of classes.simulators.pytorch.module_profiler.module_shape_profiler() function
    If there are two layers with the same name module_shape_profiler() returns a bad output and an error will
    be raised during error simulation.


    Args
    ----
    * `module : nn.Module`. The module to be profiled
    * `input_data : Tensor | None`. A dummy input accepted from the module
    * `input_shape : Tensor | None`. The shape of an input tensor accepted by the network. This must be specified only if `input_data` is not specified.
    * `device`: The torch device where the test inference is executed. If not specified defaults to cuda if available, otherwise cpu. Note that the model
                will be moved to device as a side-effect of this function.
    * `module_filter_fn : Callable[[str, nn.Module], bool]`. A function that takes in input the module name and the module itself and returns a boolean that says
                whether the profiling should happen in that layer. If not specified, the output shape of all modules will be profiled.
    * `profile_input_shapes : boolean`. If False the shape profiled will be output shapes. If True, the function profiles modules input shapes. Defaults to False.
    Returns
    ---
    A dictionary that has the submodules fully qualified names as keys and their corresponding input or output shapes
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
            if profile_input_shapes:
                if isinstance(input[0], torch.Tensor):
                    shape_index[name] = input[0].size()
            else:
                if isinstance(output, torch.Tensor):
                    shape_index[name] = output.size()
            # Do not modify the output
            return output

        return _shape_profile_hook

    if input_data is not None and input_shape is None:
        input_shape = input_data.shape
        input_data = input_data.to(device)
    elif input_data is None and input_shape is not None:
        input_data = torch.normal(0.0, 1.0, input_shape).to(device)
    else:
        raise ValueError(
            "One and only one between input_data and input_shape must be specified."
        )

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
    module_filter_fn: Callable[[str, nn.Module], bool] = lambda module, name: True,
    device=DEFAULT_DEVICE,
) -> Mapping[str, np.ndarray]:

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

    profiled_modules_names = []

    try:
        for name, module in network.named_modules():
            if module_filter_fn(name, module):
                profiled_modules_names.append(name)
                handle = module.register_forward_hook(_make_range_profile_hook(name))
                hook_handles.append(handle)
        with torch.no_grad():
            for data in tqdm(dataloader, desc="Range Profiling"):
                network_input = network_input_fn(data)
                network_input = network_input.to(device)
                output = network(network_input)

    finally:
        for handle in hook_handles:
            handle.remove()

    result = {}

    for module_name in profiled_modules_names:
        min_val = min_value_per_module[module_name].cpu().numpy()
        max_val = max_value_per_module[module_name].cpu().numpy()
        result[module_name] = np.array([min_val, max_val], dtype=np_output_dtype)

    return result


def save_range_profile_to_json(
    file_path: str, range_profile: Mapping[str, np.ndarray], indent=2
):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    range_dict = {mod: rng.tolist() for mod, rng in range_profile.items()}
    with open(file_path, "w") as f:
        json.dump(range_dict, f, indent=indent)


def generate_and_persist_range_profile(
    file_path: str,
    network: nn.Module,
    dataloader: DataLoader,
    network_input_fn: Callable = itemgetter(0),
    torch_dtype=torch.float32,
    np_output_dtype=np.float32,
    module_filter_fn: Callable[[str, nn.Module], bool] = lambda module, name: True,
    indent=2,
    device=DEFAULT_DEVICE,
    exists_ok=True,
    overwrite=False
) -> Mapping[str, np.ndarray]:
    if os.path.exists(file_path) and not overwrite:
        if not exists_ok:
            raise FileExistsError(f'Range profile at {file_path} already exists and exists_ok is False.')
        with open(file_path, 'r') as f:
            profile_dict = json.load(f)
            range_profile = {mod: np.array(rng, dtype=np_output_dtype) for mod, rng in profile_dict.items()}
            return range_profile
    else:
        range_profile = module_range_profiler(
            network,
            dataloader,
            network_input_fn,
            torch_dtype,
            np_output_dtype,
            module_filter_fn,
            device
        )
        save_range_profile_to_json(file_path, range_profile, indent)
        return range_profile
    

def profile_module_execution_time(
    network : nn.Module,
    input_data : Optional[torch.Tensor] = None,
    input_shape : Optional[Sequence[int]] = None,
    module_filter_fn: Callable[[str, nn.Module], bool] = lambda module, name: True,
    warmup_runs : int = 10,
    profile_runs : int = 100,
    device=DEFAULT_DEVICE

) -> Mapping[str, Tuple[float, float]]:
    
    
    network.to(device)
    if input_data is None and input_shape is None or (input_data is not None and input_shape is not None):
        raise ValueError('One and only one argument between input_data and input_shape must be specified. here is the entier bee movie script')
    
    if input_data is None and input_shape is not None:
        input_data = torch.normal(0.0, 1.0, size=input_shape)
    elif input_data is not None and input_shape is None:
        input_shape = input_data.size()
    else:
        raise ValueError('Something really strange happened. Execution should not end up here.')
    
    input_data = input_data.to(device)
    # Get input shapes for reproducing inputs for the module
    input_shapes = module_shape_profiler(
        network, 
        input_data=input_data, 
        device=device, 
        module_filter_fn=module_filter_fn,
        profile_input_shapes=True)
    
    runtimes_ms = {}

    modules_to_profile = [(mod_name, module) for mod_name, module in network.named_modules() if module_filter_fn(mod_name, module)]
    for mod_name, module in tqdm(modules_to_profile, desc='Time Profiling'):
        
        if module_filter_fn(mod_name, module): 
            if mod_name not in input_shapes:
                print(f'WARNING: {mod_name} shape not found in shape profile index. Skipping it.')
            else:
                module_input_shape = input_shapes[mod_name]
                # Warmup Runs
                for _ in range(warmup_runs): 
                    module_input = torch.normal(0.0, 1.0, size=module_input_shape).to(device)
                    module(module_input)
                # Effective Runs
                times_ms = np.zeros(profile_runs)
                for i in range(profile_runs):
                    module_input = torch.normal(0.0, 1.0, size=module_input_shape).to(device)

                    torch.cuda.empty_cache()  # Clear CUDA cache
                    gc.collect()  # Collect garbage to prevent interference
                    torch.cuda.synchronize()  # Synchronize to ensure starting on a clean slate

                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    module(module_input)
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    times_ms[i] = start_event.elapsed_time(end_event)
                runtimes_ms[mod_name] = (times_ms.mean().item(), times_ms.std().item())
    if len(runtimes_ms) == 0:
        raise RuntimeError('No modules profiled')
    return runtimes_ms

def generate_and_persist_execution_time_profile(
    file_path: str,
    module : nn.Module,
    input_data : Optional[torch.Tensor] = None,
    input_shape : Optional[Sequence[int]] = None,
    module_filter_fn: Callable[[str, nn.Module], bool] = lambda module, name: True,
    warmup_runs : int = 10,
    profile_runs : int = 100,
    device=DEFAULT_DEVICE,
    exists_ok=True,
    overwrite=False
) -> Mapping[str, Tuple[float, float]]:
    if os.path.exists(file_path) and not overwrite:
        if not exists_ok:
            raise FileExistsError(f'Range profile at {file_path} already exists and exists_ok is False.')
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            reader_iter = iter(reader)
            headers = next(reader_iter)
            result = {}
            for row in reader_iter:
                mod_name, time, std = row
                result[mod_name] = (time, std)
            return result
    else:
        profile = profile_module_execution_time(
            module,
            input_data=input_data,
            input_shape=input_shape,
            warmup_runs=warmup_runs,
            profile_runs=profile_runs,
            module_filter_fn=module_filter_fn,
            device=device
        )
        with open(file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['module_name', 'avg_time_ms', 'std_dev_ms'])
            for mod, (avg_time_ms, std_dev_ms) in profile.items():
                writer.writerow([mod, avg_time_ms, std_dev_ms])
        return profile
    

