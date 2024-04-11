from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import tarfile
import json
import shutil

from torch.utils.hooks import RemovableHandle

import os
import math

from typing import Any, Callable, Generator, List, Mapping, Optional, Sequence, Tuple

from tqdm import tqdm

from classes.fault_generator.fault import FaultBatch
from classes.simulators.pytorch.error_model_mapper import (
    ModuleToFaultGeneratorMapper,
    create_module_to_generator_mapper,
)
from classes.simulators.pytorch.module_profiler import module_shape_profiler


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PyTorchFaultListInfo:
    input_shape : Sequence[int]
    batch_dimension : Optional[int]
    modules_output_shapes : Mapping[str, Sequence[int]]
    n_faults_per_module : int
    fault_batch_size : int
    injectable_layers : Sequence[str]

    @classmethod
    def load_fault_list_info(cls, fault_list_path):
        with tarfile.TarFile(fault_list_path, 'r') as tarf:
            member_file = tarf.extractfile('fault_list.json')
            if not member_file:
                raise FileNotFoundError(f'Fault List descriptor file fault_list.json not found in fault list archive')
            try:
                fault_list_info = json.load(member_file)
                input_shape = fault_list_info['input_shape']
                batch_dimension = fault_list_info['batch_dimension']
                modules_output_shapes = fault_list_info['modules_output_shapes']
                n_faults_per_module = fault_list_info['n_faults_per_module']
                fault_batch_size = fault_list_info['fault_batch_size']
                injectable_layers = fault_list_info['injectable_layers']
                return cls(input_shape, batch_dimension, modules_output_shapes, n_faults_per_module, fault_batch_size, injectable_layers)
            finally:
                member_file.close()

    def to_dict(self) -> Mapping[str, Any]:
        return {
                "input_shape": self.input_shape,
                "batch_dimension": self.batch_dimension,
                "modules_output_shapes": self.modules_output_shapes,
                "n_faults_per_module": self.n_faults_per_module,
                "injectable_layers": self.injectable_layers,
                "fault_batch_size": self.fault_batch_size,
            }

class PyTorchFaultList:
    def __init__(
        self,
        network: nn.Module,
        input_shape: Optional[Sequence[int]] = None,
        input_data: Optional[torch.Tensor] = None,
        module_to_fault_generator_fn: ModuleToFaultGeneratorMapper = create_module_to_generator_mapper(),
        batch_axis: Optional[int] = 0,
        device=DEFAULT_DEVICE,
    ) -> None:
        
        self.network = network
        self.network.to(device)
        self.module_to_fault_generator_fn = module_to_fault_generator_fn
        self.batch_axis = batch_axis
        if input_data is not None and input_shape is None:
            self.input_shape = input_data.shape
        elif input_data is None and input_shape is not None:
            self.input_shape = input_shape
        else:
            raise ValueError("One and only one between input_data and input_shape must be specified.")

        # Profile only the layers to be injected (the ones that have a fault generator)
        profiler_filter = lambda name, mod: module_to_fault_generator_fn(name, mod) is not None


        self.modules_output_shapes = module_shape_profiler(
            self.network, 
            input_data, 
            input_shape, device, 
            profiler_filter
        )

        self.injectable_layers = self.get_injectable_layers_names()
        self.num_injectable_layers = len(self.injectable_layers)

    def get_injectable_layers_names(self) -> List[str]:
        names = []
        for name, module in self.network.named_modules():
            fault_generator = self.module_to_fault_generator_fn(name, module)
            if fault_generator:
                names.append(name)
        return names
    
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
            if self.batch_axis is not None: 
                output_shape.pop(self.batch_axis)
            masks, values, values_index, sp_classes, sp_parameters = fault_generator.generate_batched_mask(
                output_shape, fault_batch_size
            )
            yield module_name, it, FaultBatch(masks, values, values_index, sp_classes, sp_parameters)

    def network_fault_list_generator(
        self, n_faults: int, fault_batch_size: int = 1
    ) -> Generator[Tuple, Any, None]:

        # To generalize this class to be used keras we should generalize the iterator for the layers of the network
        # You can use keras.Model._flatten_layers model to iterate trough all layers of the model
        # recursively, and then for getting the name you can access the layer .name attribute
        # https://github.com/keras-team/keras/blob/v3.1.1/keras/layers/layer.py#L1343
        for name, module in self.network.named_modules():
            yield from self.module_fault_list_generator(name, n_faults, fault_batch_size)
    

    def create_and_save_fault_list(
        self,
        output_path: str,
        n_faults: int,
        fault_batch_size: int = 1,
        show_progress=True,
        exists_ok=True,
        overwrite=False
    ):

        if n_faults % fault_batch_size != 0:
            raise ValueError(f'Number of faults per modules (n_faults) must be multiple of fault_batch size. Instead found {n_faults=} {fault_batch_size=}')
        
        if output_path.endswith('.tar'):
            temp_output_dir = output_path[:-4] + '_tmp'
        else:
            temp_output_dir = output_path + '_tmp'
            output_path = output_path + '.tar'

        if os.path.exists(output_path):
            if not exists_ok:
                raise FileExistsError(f'FaultList already exists at {output_path}')
            if not overwrite:
                return


        try:
            os.makedirs(temp_output_dir, exist_ok=False)
            count = 0
            n_iters = int(math.ceil(n_faults / fault_batch_size)) * self.num_injectable_layers

            pbar = None
            if show_progress:
                pbar = tqdm(total=n_iters)
            
            self.info = PyTorchFaultListInfo(
                input_shape=list(self.input_shape),
                batch_dimension=self.batch_axis,
                modules_output_shapes=self.modules_output_shapes,
                n_faults_per_module=n_faults,
                injectable_layers=self.injectable_layers,
                fault_batch_size=fault_batch_size,
            )

            with open(os.path.join(temp_output_dir, 'fault_list.json'),'w') as f:
                json.dump(self.info.to_dict(), f)


            for name, module in self.network.named_modules():
                for module_name, batch_num, fault_batch in self.module_fault_list_generator(name, n_faults, fault_batch_size):
                    if pbar:
                        pbar.set_description(module_name)
                    module_path = os.path.join(temp_output_dir, module_name)
                    os.makedirs(module_path, exist_ok=True)

                    file_name = os.path.join(module_path, f"{batch_num}_faults_{module_name}.npz")
                    npz_dict = {
                        "masks": fault_batch.corrupted_value_mask,
                        "values": fault_batch.corrupted_values,
                        "values_index": fault_batch.corrupted_values_index,
                        "sp_classes": fault_batch.spatial_pattern_names,
                        "sp_parameters": fault_batch.sp_parameters,
                        "module": np.asarray(module_name),
                        "seq": np.int64(count),
                    }
                    np.savez_compressed(file_name, **npz_dict)
                    count += 1
                    if pbar:
                        pbar.update(1)

            with tarfile.TarFile(output_path, 'w') as tarf:
                tarf.add(os.path.join(temp_output_dir, 'fault_list.json'), arcname='fault_list.json')
                for module_folder in os.listdir(temp_output_dir):
                    module_folder_path = os.path.join(temp_output_dir, module_folder)
                    if os.path.isdir(module_folder_path):
                        for batch_file in os.listdir(module_folder_path):
                            batch_file_path = os.path.join(module_folder_path, batch_file)
                            tarf.add(batch_file_path, arcname=os.path.join(module_folder, batch_file))
        finally:
            shutil.rmtree(temp_output_dir, ignore_errors=True)
