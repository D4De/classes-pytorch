import numpy as np
import torch
import torch.nn as nn

from torch.utils.hooks import RemovableHandle

import os
import math

from typing import Any, Callable, Generator, Optional, Sequence, Tuple

from tqdm import tqdm

from classes.fault_generator.fault_generator import FaultGenerator
from classes.simulators.pytorch.error_model_mapper import (
    ModuleToFaultGeneratorMapper,
    create_module_to_generator_mapper,
)
from classes.simulators.pytorch.network_profiler import network_shape_profiler


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class FaultListGenerator:
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
        if input_data:
            self.input_shape = input_data.shape

        self.shape_index = network_shape_profiler(
            self.network, input_data, input_shape, device
        )

        self.injectable_layers = self._count_injectable_layers()

    def _count_injectable_layers(self) -> int:
        count = 0
        for name, module in self.network.named_modules():
            fault_generator = self.module_to_fault_generator_fn(name, module)
            if fault_generator:
                count += 1
        return count

    def network_fault_list_generator(
        self, n_faults: int, fault_batch_size: int = 1
    ) -> Generator[Tuple, Any, None]:

        n_iters = int(math.ceil(n_faults / fault_batch_size))

        for name, module in self.network.named_modules():
            fault_generator = self.module_to_fault_generator_fn(name, module)
            if fault_generator:
                for it in range(n_iters):
                    output_shape = list(self.shape_index[name])
                    if self.batch_dimension:
                        del output_shape[self.batch_dimension]
                    masks, values, values_index = fault_generator.generate_batched_mask(
                        output_shape, fault_batch_size
                    )
                    yield name, (masks, values, values_index)

    def save_fault_list_to_files(
        self,
        dir_path: str,
        n_faults: int,
        fault_batch_size: int = 1,
        show_progress=True,
    ):
        os.makedirs(dir_path, exist_ok=True)
        count = 0
        n_iters = int(math.ceil(n_faults / fault_batch_size)) * self.injectable_layers

        pbar = None
        if show_progress:
            pbar = tqdm(total=n_iters)
        
        for module_name, (masks, values, values_index) in self.network_fault_list_generator(n_faults, fault_batch_size):
            if pbar:
                pbar.set_description(module_name)
            file_name = os.path.join(dir_path, "faults_{module_name}_seq_{count}.npz")
            npz_dict = {
                "masks": masks,
                "values": values,
                "values_index": values_index,
                "module": np.asarray(module_name),
                "seq": np.int64(count),
            }
            np.savez_compressed(file_name, **npz_dict)
            count += 1
            if pbar:
                pbar.update(1)
