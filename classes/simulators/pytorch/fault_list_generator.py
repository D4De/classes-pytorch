import numpy as np
import torch
import torch.nn as nn

from torch.utils.hooks import RemovableHandle

import os
import math

from typing import Any, Callable, Generator, List, Literal, Optional, Sequence, Tuple, Union

from tqdm import tqdm

from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.simulators.pytorch.network_profiler import network_shape_profiler


DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FaultListGenerator:

    def __init__(
            self, 
            network : nn.Module,
            output_folder_path : str,
            error_model_mapper_f : Callable[[str, nn.Module], Optional[FaultGenerator]],
            input_shape : Optional[Sequence[int]] = None,
            input_data : Optional[torch.Tensor] = None,
            batch_dimension : Optional[int] = 0,
            device = DEFAULT_DEVICE
        ) -> None:

        self.network = network
        self.network.to(device)
        self.error_model_mapper_f = error_model_mapper_f
        self.batch_dimension = batch_dimension
        if input_data:
            self.input_shape = input_data.shape

        self.shape_index = network_shape_profiler(
            self.network,
            input_data,
            input_shape,
            device
        )

        self.injectable_layers = self._count_injectable_layers()

    def _count_injectable_layers(self) -> int:
        count = 0
        for name, module in self.network.named_modules():
            fault_generator = self.error_model_mapper_f(name, module)
            if fault_generator:
                count += 1
        return count

    def network_fault_list_generator(
        self,
        n_faults : int,
        fault_batch_size : int = 1
    ) -> Generator[Tuple, Any, None]:

        n_iters = int(math.ceil(n_faults / fault_batch_size))

        for name, module in self.network.named_modules():
            fault_generator = self.error_model_mapper_f(name, module)
            if fault_generator:
                for it in range(n_iters):
                    output_shape = list(self.shape_index[name])
                    if self.batch_dimension:
                        del output_shape[self.batch_dimension]
                    masks, values, values_index = fault_generator.generate_batched_mask(
                        output_shape,
                        fault_batch_size
                    )
                    yield name, (masks, values, values_index)
    
    def save_fault_list_to_files(
        self,
        dir_path : str,
        n_faults : int,
        fault_batch_size : int = 1,
        show_progress = True
    ):
        os.makedirs(dir_path, exist_ok=True)
        count = 0
        iterator = self.network_fault_list_generator(n_faults, fault_batch_size)
        n_iters = int(math.ceil(n_faults / fault_batch_size)) * self.injectable_layers

        if show_progress:
            pbar = tqdm(total=n_iters)
        for module_name, (masks, values, values_index) in iterator:
            file_name = os.path.join(dir_path, 'faults_{module_name}_seq_{count}.npz')
            npz_dict = {'masks': masks, 'values': values, 'values_index': values_index, 'module': np.asarray(module_name), 'seq': np.int64(count)}
            np.savez_compressed(file_name, **npz_dict)
            count += 1
            if show_progress:
                pbar.update(1)
    

