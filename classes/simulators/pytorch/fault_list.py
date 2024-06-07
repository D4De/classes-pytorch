from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import tarfile
import json
import shutil
import warnings

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
class PyTorchFaultListMetadata:
    """
    Contains the metadata of a Fault list generated for a PyTorch model.
    """

    input_shape: Sequence[int]
    """
    The input shape of the PyTorch model    
    """

    batch_dimension: Optional[int]
    """
    The dimension of the batch.
    """
    modules_output_shapes: Mapping[str, Sequence[int]]
    n_faults_per_module: int
    fault_batch_size: int
    injectable_layers: Sequence[str]

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
                input_shape = fault_list_info["input_shape"]
                batch_dimension = fault_list_info["batch_dimension"]
                modules_output_shapes = fault_list_info["modules_output_shapes"]
                n_faults_per_module = fault_list_info["n_faults_per_module"]
                fault_batch_size = fault_list_info["fault_batch_size"]
                injectable_layers = fault_list_info["injectable_layers"]
                return cls(
                    input_shape,
                    batch_dimension,
                    modules_output_shapes,
                    n_faults_per_module,
                    fault_batch_size,
                    injectable_layers,
                )
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
    """
    A PyTorch fault list 
    """
    def __init__(
        self,
        network: nn.Module,
        input_shape: Optional[Sequence[int]] = None,
        input_data: Optional[torch.Tensor] = None,
        module_to_fault_generator_fn: ModuleToFaultGeneratorMapper = create_module_to_generator_mapper(),
        module_to_range_map : Optional[Mapping[str, np.ndarray]] = None,
        default_range : np.ndarray = np.array([-30.0, 30.0],dtype=np.float32),
        value_dtype : type = np.float32,
        batch_axis: Optional[int] = 0,
        device=DEFAULT_DEVICE,
    ) -> None:

        self.network = network
        self.network.to(device)
        self.module_to_fault_generator_fn = module_to_fault_generator_fn
        self.module_to_range_map = module_to_range_map
        self.default_range = default_range
        self.value_dtype= value_dtype
        self.batch_axis = batch_axis
        if input_data is not None and input_shape is None:
            self.input_shape = input_data.shape
        elif input_data is None and input_shape is not None:
            self.input_shape = input_shape
        else:
            raise ValueError(
                "One and only one between input_data and input_shape must be specified."
            )

        # Profile only the layers to be injected (the ones that have a fault generator)
        profiler_filter = (
            lambda name, mod: module_to_fault_generator_fn(name, mod) is not None
        )

        self.modules_output_shapes = module_shape_profiler(
            self.network, input_data, input_shape, device, profiler_filter
        )

        self.injectable_layers = self.get_injectable_submodules_names()
        self.num_injectable_layers = len(self.injectable_layers)

    def get_injectable_submodules_names(self) -> List[str]:
        """
        Gets the list of all the names of all submodules layers of the target network.

        A submodules is injectable if and only if the ``module_to_fault_generator_fn`` tied to this
        instance returns a ``FaultGenerator`` when called on that submodule.

        Returns
        ---
        A list of strings containing the fully qualified names of all the injectable layers of the network.
        """
        names = []
        for name, module in self.network.named_modules():
            fault_generator = self.module_to_fault_generator_fn(name, module)
            if fault_generator:
                names.append(name)
        return names

    def module_fault_list_generator(
        self, module_name: str, n_faults: int, fault_batch_size: int = 1
    ):
        """
        Creates a generator that generates lazily a finite number of faults faults for a specific module.
        The faults are generated using the fault generator related to the module accordingly to
        the ``module_to_fault_generator_fn`` tied to this instance.

        Args
        ---
        * ``module_name : str``: The complete, fully qualifed name of the submodule to generate faults for.
        * ``n_faults: int.``: Number of faults to generate.
        * ``fault_batch_size: int``: The size of each group (batch) of faults generated at once.

        Returns
        ---
        A generator object that could be iterated through to generate lazily faults.
        Each iteration returns a tuple of three elements:
        * The name of the module itself (equal to ``module_name`` every time)
        * The number of the iteration
        * A FaultBatch object containing ``fault_batch_size`` faults.

        The generator stops after generating ``n_faults``, and will be empty if there is no
        fault generator object associated to the module.

        Raises
        ---
        * ``AttributeError``: If ``module_name`` references an invalid
        path to a module or resolves to something that is not an
        ``nn.Module``.
        """

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
            if self.module_to_range_map is not None:
                operating_range = self.module_to_range_map.get(module_name) 
                if operating_range is None:
                    operating_range = self.default_range
            else:
                operating_range = self.default_range

            fault_batch = (
                fault_generator.generate_batched_mask(
                    output_shape, 
                    fault_batch_size, 
                    value_range=operating_range, 
                    dtype=self.value_dtype)
            )
            yield module_name, it, fault_batch

    def network_fault_list_generator(
        self, n_faults: int, fault_batch_size: int = 1
    ) -> Generator[Tuple, Any, None]:
        """
        Creates a generator that generates lazily a finite number of faults faults for each injectable module of the network.
        A module is injectable if it is associated to a ``FaultGenerator`` object returned by
        the ``module_to_fault_generator_fn`` tied to this instance.

        The generator traverses all the submodules with the order of the ``torch.nn.Module.named_modules`` method.

        Args
        ---
        * ``n_faults: int.``: Number of faults to generate for each module.
        * ``fault_batch_size: int``: The size of each group (batch) of faults generated at once.

        Returns
        ---
        A generator object that could be iterated through to generate lazily faults.
        Each iteration returns a tuple of three elements:
        * The name of the module for which the fault was generated
        * The number of the batch relative to the current submodule. The numbering restarts for every submodule.
        * A FaultBatch object containing ``fault_batch_size`` faults.

        The generator stops after generating ``n_faults`` for each injectable submodule, and will skip
        non-injectable modules.
        """
        # To generalize this class to be used keras we should generalize the iterator for the layers of the network
        # You can use keras.Model._flatten_layers model to iterate trough all layers of the model
        # recursively, and then for getting the name you can access the layer .name attribute
        # https://github.com/keras-team/keras/blob/v3.1.1/keras/layers/layer.py#L1343
        for name, module in self.network.named_modules():
            yield from self.module_fault_list_generator(
                name, n_faults, fault_batch_size
            )

    def generate_and_persist_fault_list(
        self,
        output_path: str,
        n_faults_per_module: int,
        fault_batch_size: int = 1,
        show_progress=True,
        exists_ok=True,
        overwrite=False,
    ):
        """
        Generates the faults of the faultlist according to the error models and saves the faults to be used later in an error simulation campaign.
        Persists all those faults of the faultlist into a tar file that can be loaded using the ``FaultListFromTarFile``, similarly to how
        a traditional PyTorch input dataset is loaded.
        The faults are not all stored in RAM and saved all once but are rather saved in a streamed fashion

        The output tar file is structured as it follows:
        - ``fault_list.json``: A json file containing the basic info of the fault list. This file contains the data in the info member
            of this class, of type PyTorchFaultListInfo.
        - For each module of the model there is a folder containing various .npz files, each one contains a batch of faults of size ``fault_batch_size``.
          The files in the module folder are named with this format ``"{batch_num}_faults_{module_name}.npz"``

        Args
        ---
        * ``output_path : str``. Path where the output tar file is saved. Note that while saving the file a temp folder could be created for storing temporarly the
            produced files.
        * ``n_faults : int``. Number of faults to generate for each module of the network. The total number of faults generated is simply ``n_faults * n_modules``.
        * ``fault_batch_size : int``. The size of the batch stored in each npz file. For example 128 faults with a batch size of 8 are saved in 16 files. It must
            be a divisor of ``n_faults``. This parameters affects heavily the size of the final archive, so in presence of disk space constraints
            it should be chosen accurately.
        * ``show_progress : bool``. Draws a progress bar when generating and saving faults. Defaults to ``True``.
        * ``exists_okay : bool``. If ``False`` a FileExistsError will be thrown if a fault list with the same output path already exists. Defaults to ``True``.
        * ``overwrite : bool``. If ``False`` the function will not generate a fault list and gracefully return if a fault list with the same output path already exists. Defaults to ``True``.

        Returns
        ---
        ``None``

        Raises
        ---
        * ``ValueError``. When ``fault_batch_size`` does not divide ``n_faults``
        * ``FileExistsError``. If ``exists_ok`` is ``False`` and a fault list at ``output_path`` already exists. This error is raised regardless of the value of ``overwrite``.
        """

        if n_faults_per_module % fault_batch_size != 0:
            raise ValueError(
                f"Number of faults per modules (n_faults) must be multiple of fault_batch size. Instead found {n_faults_per_module=} {fault_batch_size=}"
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
            count = 0
            n_iters = (
                int(math.ceil(n_faults_per_module / fault_batch_size))
                * self.num_injectable_layers
            )

            pbar = None
            if show_progress:
                pbar = tqdm(total=n_iters)
            # Create info object and be store it as the fault list descriptor
            info = PyTorchFaultListMetadata(
                input_shape=list(self.input_shape),
                batch_dimension=self.batch_axis,
                modules_output_shapes=self.modules_output_shapes,
                n_faults_per_module=n_faults_per_module,
                fault_batch_size=fault_batch_size,
                injectable_layers=self.injectable_layers,
            )

            with open(os.path.join(temp_output_dir, "fault_list.json"), "w") as f:
                json.dump(info.to_dict(), f)

            for name, module in self.network.named_modules():
                # Generate faults lazily and save them to the temp directory
                for (
                    module_name,
                    batch_num,
                    fault_batch,
                ) in self.module_fault_list_generator(
                    name, n_faults_per_module, fault_batch_size
                ):
                    if pbar:
                        pbar.set_description(f"Generating Faults")
                        pbar.set_postfix_str(module_name)
                    module_path = os.path.join(temp_output_dir, module_name)
                    os.makedirs(module_path, exist_ok=True)

                    file_name = os.path.join(
                        module_path, f"{batch_num}_faults_{module_name}.npz"
                    )
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
            # Archive all files
            with tarfile.TarFile(output_path, "w") as tarf:
                tarf.add(
                    os.path.join(temp_output_dir, "fault_list.json"),
                    arcname="fault_list.json",
                )
                for module_folder in os.listdir(temp_output_dir):
                    module_folder_path = os.path.join(temp_output_dir, module_folder)
                    if os.path.isdir(module_folder_path):
                        for batch_file in os.listdir(module_folder_path):
                            batch_file_path = os.path.join(
                                module_folder_path, batch_file
                            )
                            tarf.add(
                                batch_file_path,
                                arcname=os.path.join(module_folder, batch_file),
                            )
        finally:
            # Delete in any case the temp directory, even when a crash happens
            shutil.rmtree(temp_output_dir, ignore_errors=True)
