import os
import json
import math
import numpy as np
import torch
import shutil
import tarfile

from PIL import Image
from typing import Any
from dataclasses import dataclass

from classes.fault_generator.fault_generator import FaultGenerator

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#------------------------------------------------------------
# Rework of the fault list to account for multiple error models

@dataclass
class PyTorchFaultListDynamicMetadata:
    input_shape         : torch.Size | list[int]
    batch_dimension     : int | None
    module_shapes       : dict[str, tuple[list[int], list[int]]]
    num_module_faults   : list[int]
    fault_batch_size    : int
    injectable_layers   : list[str]

    @classmethod
    def load_fault_list_info(cls, fault_list_path):
        with tarfile.TarFile(fault_list_path, "r") as tarf:
            member_file = tarf.extractfile("fault_list.json")
            if not member_file:
                raise FileNotFoundError(
                    f"Fault List descriptor file fault_list.json not found in fault list archive"
                )
            try:
                fault_list_info = json.load(member_file)

                input_shape             = fault_list_info["input_shape"]
                batch_dimension         = fault_list_info["batch_dimension"]
                module_shapes           = fault_list_info["module_shapes"]
                num_module_faults     = fault_list_info["num_module_faults"]
                fault_batch_size        = fault_list_info["fault_batch_size"]
                injectable_layers       = fault_list_info["injectable_layers"]
                return cls(
                    input_shape,
                    batch_dimension,
                    module_shapes,
                    num_module_faults,
                    fault_batch_size,
                    injectable_layers,
                )
            finally:
                member_file.close()

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_shape"           : self.input_shape,
            "batch_dimension"       : self.batch_dimension,
            "module_shapes"         : self.module_shapes,
            "num_module_faults"   : self.num_module_faults,
            "injectable_layers"     : self.injectable_layers,
            "fault_batch_size"      : self.fault_batch_size,
        }


class PyTorchFaultListDynamic:
    def __init__(
        self,
        module_names: list[str],
        module_shapes: dict,
        module_ranges: dict,
        module_to_fault_generator: dict[str, FaultGenerator],
        input_sample: torch.Tensor | tuple[Image.Image],
        logger,
        value_dtype : type = np.float32,
        batch_axis: int | None = 0,
    ):
        logger.info('Creating fault list...')
        self.module_names              = module_names
        self.module_shapes             = module_shapes
        self.module_ranges             = module_ranges
        self.module_to_fault_generator = module_to_fault_generator
        self.value_dtype               = value_dtype
        self.batch_axis                = batch_axis

        # record the input shape
        if isinstance(input_sample, torch.Tensor):
            self.input_shape = input_sample.shape
        elif isinstance(input_sample, tuple): # tuple of PIL images
            width, height = input_sample[0].size
            self.input_shape = torch.Size([len(input_sample), 3, height, width]) # build input shape according to PIL Image shape
        else:
            raise ValueError(f'Unsupported type of input sample: {type(input_sample)}')
        

    def module_fault_list_generator(
        self,
        module_name: str,
        num_faults: int,
        uniform_spatial_classes: bool,
        fault_batch_size: int = 1,
        force_single_channel=False,
    ):
        num_batches = int(math.ceil(num_faults / fault_batch_size))
        fault_generator = self.module_to_fault_generator[module_name]

        output_shape = list(self.module_shapes[module_name][0]).copy()
        if self.batch_axis is not None:
            output_shape.pop(self.batch_axis)

        operating_range = self.module_ranges[module_name]

        if uniform_spatial_classes:
            current_batch = 0
            # get the spatial classes of the error model and generate num_batches batches for each one
            for spatial_class_name in fault_generator.get_available_spatial_classes():
                for it in range(current_batch, current_batch + num_batches):
                    fault_batch = fault_generator.generate_batched_mask(
                        output_shape, 
                        fault_batch_size,
                        spatial_class = spatial_class_name,
                        value_range = operating_range, 
                        dtype = self.value_dtype,
                        force_single_channel=False,
                    )
            
                    yield it, fault_batch
                
                current_batch += num_batches

        else:
            # generate batches with random spatial classes
            for it in range(num_batches):
                fault_batch = fault_generator.generate_batched_mask(
                    output_shape, 
                    fault_batch_size,
                    spatial_class = None,
                    value_range = operating_range, 
                    dtype = self.value_dtype
                )
            
                yield it, fault_batch


    def generate_and_persist_fault_list(
        self,
        output_path: str,
        n_faults_per_module: int,
        uniform_spatial_classes: bool,
        logger,
        fault_batch_size: int = 1,
        exists_ok=True,
        overwrite=False,
        force_single_channel=False,
    ):
        logger.info('Started fault generation...')
        if n_faults_per_module % fault_batch_size != 0:
            raise ValueError(
                f"Number of faults per modules must be multiple of fault_batch size. Instead found {n_faults_per_module=} {fault_batch_size=}"
            )

        # Remove .tar from temp dir if present, add it to the output path if not present
        if output_path.endswith(".tar"):
            temp_output_dir = output_path[:-4] + "_tmp"
        else:
            temp_output_dir = output_path + "_tmp"
            output_path = output_path + ".tar"

        if os.path.exists(output_path):
            if not exists_ok:
                raise FileExistsError(f"FaultList already exists at {output_path}")
            if not overwrite:
                return

        try:
            os.makedirs(temp_output_dir, exist_ok=False)

            num_module_faults: list[int] = [] # track how many faults are generated per module
            total_batch_counter = 0

            for module_name in self.module_names:
                logger.info(f'Generating faults for layer {module_name}')
                module_batch_counter = 0

                # Generate faults lazily and save them to the temp directory
                for batch_num, fault_batch in self.module_fault_list_generator(
                    module_name, n_faults_per_module, uniform_spatial_classes, fault_batch_size, force_single_channel=False,
                ):
                    module_path = os.path.join(temp_output_dir, module_name)
                    os.makedirs(module_path, exist_ok=True)

                    file_name = os.path.join(
                        module_path, f"{batch_num}_faults_{module_name}.npz"
                    )
                    npz_dict = {
                        "masks"         : fault_batch.corrupted_value_mask,
                        "values"        : fault_batch.corrupted_values,
                        "values_index"  : fault_batch.corrupted_values_index,
                        "sp_classes"    : fault_batch.spatial_pattern_names,
                        "sp_parameters" : fault_batch.sp_parameters,
                        "module"        : np.asarray(module_name),
                        "seq"           : np.int64(total_batch_counter),
                    }
                    np.savez_compressed(file_name, **npz_dict)

                    module_batch_counter += 1
                    total_batch_counter += 1

                num_module_faults.append(module_batch_counter)

            # Create info object and store it as the fault list descriptor
            logger.info('Creating fault list metadata.')
            info = PyTorchFaultListDynamicMetadata(
                input_shape         = list(self.input_shape),
                batch_dimension     = self.batch_axis,
                module_shapes       = self.module_shapes,
                num_module_faults   = num_module_faults,
                fault_batch_size    = fault_batch_size,
                injectable_layers   = self.module_names,
            )

            with open(os.path.join(temp_output_dir, "fault_list.json"), "w") as f:
                json.dump(info.to_dict(), f)


            # Archive all files
            logger.info('Saving fault list to archive.')
            with tarfile.TarFile(output_path, "w") as tarf:
                tarf.add(
                    os.path.join(temp_output_dir, "fault_list.json"),
                    arcname="fault_list.json",
                )
                for module_folder in os.listdir(temp_output_dir):
                    module_folder_path = os.path.join(temp_output_dir, module_folder)

                    if os.path.isdir(module_folder_path):
                        for batch_file in os.listdir(module_folder_path):
                            batch_file_path = os.path.join(module_folder_path, batch_file)
                            tarf.add(batch_file_path, arcname=os.path.join(module_folder, batch_file))
        finally:
            # Delete in any case the temp directory, even when a crash happens
            shutil.rmtree(temp_output_dir, ignore_errors=True)