import numpy as np
import torch
import torch.nn as nn
import zipfile
import json
import shutil

from torch.utils.hooks import RemovableHandle

import os
import math

from typing import Any, Callable, Generator, Optional, Sequence, Tuple

from tqdm import tqdm

from classes.simulators.pytorch.error_model_mapper import (
    ModuleToFaultGeneratorMapper,
    create_module_to_generator_mapper,
)
from classes.simulators.pytorch.network_profiler import network_shape_profiler



DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PyTorchFaultListGenerator:
    def __init__(
        self,
        network: nn.Module,
        input_shape: Optional[Sequence[int]] = None,
        input_data: Optional[torch.Tensor] = None,
        module_to_fault_generator_fn: ModuleToFaultGeneratorMapper = create_module_to_generator_mapper(),
        batch_dimension: Optional[int] = 0,
        device=DEFAULT_DEVICE,
    ) -> None:
        
        self.network = network
        self.network.to(device)
        self.module_to_fault_generator_fn = module_to_fault_generator_fn
        self.batch_dimension = batch_dimension
        if input_data is not None and input_shape is None:
            self.input_shape = input_data.shape
        elif input_data is None and input_shape is not None:
            self.input_shape = input_shape
        else:
            raise ValueError("One and only one between input_data and input_shape must be specified.")


        self.modules_output_shapes = network_shape_profiler(
            self.network, input_data, input_shape, device
        )

        self.num_injectable_layers = self._count_injectable_layers()

    def _count_injectable_layers(self) -> int:
        count = 0
        for name, module in self.network.named_modules():
            fault_generator = self.module_to_fault_generator_fn(name, module)
            if fault_generator:
                count += 1
        return count
    
    def module_fault_list_generator(
        self, module_name : str, n_faults: int, fault_batch_size: int = 1
    ):
        module = self.network.get_submodule(module_name)
        n_iters = int(math.ceil(n_faults / fault_batch_size))
        fault_generator = self.module_to_fault_generator_fn(module_name, module)
        if not fault_generator:
            # No fault generator matched for this layer, return empty generator
            return
        for it in range(n_iters):
            output_shape = list(self.modules_output_shapes[module_name])
            if self.batch_dimension is not None: 
                output_shape.pop(self.batch_dimension)
            masks, values, values_index, sp_classes, sp_parameters = fault_generator.generate_batched_mask(
                output_shape, fault_batch_size
            )
            yield module_name, (masks, values, values_index, sp_classes, sp_parameters)

    def network_fault_list_generator(
        self, n_faults: int, fault_batch_size: int = 1
    ) -> Generator[Tuple, Any, None]:

        # To generalize this class to be used keras we should generalize the iterator for the layers of the network
        # You can use keras.Model._flatten_layers model to iterate trough all layers of the model
        # recursively, and then for getting the name you can access the layer .name attribute
        # https://github.com/keras-team/keras/blob/v3.1.1/keras/layers/layer.py#L1343
        for name, module in self.network.named_modules():
            yield from self.module_fault_list_generator(name, n_faults, fault_batch_size)
    

    def serialize_fault_list(
        self,
        output_path: str,
        n_faults: int,
        fault_batch_size: int = 1,
        show_progress=True,
    ):
        
        if n_faults % fault_batch_size != 0:
            raise ValueError(f'Number of faults per modules (n_faults) must be multiple of fault_batch size. Instead found {n_faults=} {fault_batch_size=}')
        
        if output_path.endswith('.zip'):
            temp_output_dir = output_path[:-4] + '_tmp'
        else:
            temp_output_dir = output_path + '_tmp'
            output_path = output_path + '.zip'

        try:
            os.makedirs(temp_output_dir, exist_ok=True)
            count = 0
            n_iters = int(math.ceil(n_faults / fault_batch_size)) * self.num_injectable_layers

            pbar = None
            if show_progress:
                pbar = tqdm(total=n_iters)
            
            fault_list_info = {
                "input_shape": list(self.input_shape),
                "batch_dimension": self.batch_dimension,
                "modules_output_shapes": self.modules_output_shapes,
                "n_faults_per_module": n_faults,
                "n_injectable_layers": self.num_injectable_layers,
                "fault_batch_size": fault_batch_size
            }

            with open(os.path.join(temp_output_dir, 'fault_list.json'),'w') as f:
                json.dump(fault_list_info, f)


            for name, module in self.network.named_modules():
                module_batch_count = 0
                for module_name, (masks, values, values_index, sp_classes, sp_parameters) in self.module_fault_list_generator(name, n_faults, fault_batch_size):
                    if pbar:
                        pbar.set_description(module_name)

                    module_path = os.path.join(temp_output_dir, module_name)
                    os.makedirs(module_path, exist_ok=True)
                    file_name = os.path.join(module_path, f"{module_batch_count}_faults_{module_name}.npz")
                    npz_dict = {
                        "masks": masks,
                        "values": values,
                        "values_index": values_index,
                        "sp_classes": sp_classes,
                        "sp_parameters": sp_parameters,
                        "module": np.asarray(module_name),
                        "seq": np.int64(count),
                    }
                    np.savez_compressed(file_name, **npz_dict)
                    count += 1
                    if pbar:
                        pbar.update(1)

            with zipfile.ZipFile(output_path, 'w') as zipf:
                zipf.write(os.path.join(temp_output_dir, 'fault_list.json'), arcname='fault_list.json')
                for module_folder in os.listdir(temp_output_dir):
                    module_folder_path = os.path.join(temp_output_dir, module_folder)
                    if os.path.isdir(module_folder_path):
                        for batch_file in os.listdir(module_folder_path):
                            batch_file_path = os.path.join(module_folder_path, batch_file)
                            zipf.write(batch_file_path, arcname=os.path.join(module_folder, batch_file))
        finally:
            shutil.rmtree(temp_output_dir, ignore_errors=True)
