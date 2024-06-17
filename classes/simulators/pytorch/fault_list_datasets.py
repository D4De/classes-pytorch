from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from classes.simulators.pytorch.fault_list import PyTorchFaultList, PyTorchFaultListMetadata
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

import tarfile
import os
import numpy as np

from classes.simulators.pytorch.pytorch_fault import PyTorchFault, PyTorchFaultBatch


def fault_collate_fn(data: Sequence[PyTorchFaultBatch]):
    names, fault_ids, masks, values, sp_classes, sp_parameters = zip(*data)

    torch_values_idxs = torch.LongTensor(
        [value_tensor.size() for value_tensor in values]
    )

    batch_masks = torch.stack(masks)
    batch_values = torch.concat(values)

    batch_values_idxs = torch.LongTensor(
        torch.zeros(len(masks) + 1).long()
    )  # sono veramente long
    batch_values_idxs[1:] = torch.cumsum(torch_values_idxs.squeeze(), dim=0)

    return PyTorchFaultBatch(
        names,
        fault_ids,
        batch_masks,
        batch_values,
        batch_values_idxs,
        sp_classes,
        sp_parameters,
    )



class FaultListFromTarFile(Dataset[PyTorchFault]):
    def __init__(self, fault_list_path: str, module: Optional[str] = None) -> None:
        super().__init__()
        self.fault_list_path = fault_list_path
        self.module = module
        self.include_sp_parameters = True
        self.fault_list_info = None
        self.tmp_path = None

        self.info = PyTorchFaultListMetadata.load_fault_list_info(self.fault_list_path)

        self.n_faults = self.info.n_faults_per_module

    def __getitem__(self, index: Union[int, Tuple[str, int]]) -> PyTorchFault:

        if isinstance(index, tuple):
            selected_module_name, module_idx = index
            if selected_module_name not in self.info.injectable_layers:
                raise IndexError(f'{selected_module_name} is not an injectable layer of the fault list.')
            
        else:
            index : int = index
            if self.module is None:
                selected_module_idx = index // self.info.n_faults_per_module
                selected_module_name = self.info.injectable_layers[selected_module_idx]
                module_idx = index % self.info.n_faults_per_module
            else:
                selected_module_name = self.module
                module_idx = index

        file_idx = module_idx // self.info.fault_batch_size
        batch_idx = module_idx % self.info.fault_batch_size
        fault_file_path = os.path.join(
            selected_module_name, f"{file_idx}_faults_{selected_module_name}.npz"
        )

        with tarfile.TarFile(self.fault_list_path, "r") as tarf:
            member_file = tarf.extractfile(fault_file_path)
            if not member_file:
                raise FileNotFoundError(
                    f"File {fault_file_path} not found in fault list archive"
                )
            try:
                fault_list_file = np.load(member_file, allow_pickle=True)
                mask = fault_list_file["masks"][batch_idx]
                slice_begin, slice_end = fault_list_file["values_index"][
                    batch_idx : batch_idx + 2
                ]
                values = fault_list_file["values"][slice_begin:slice_end]

                torch_mask = torch.from_numpy(mask)
                torch_values = torch.from_numpy(values)

                sp_class = fault_list_file["sp_classes"][batch_idx]
                sp_param = fault_list_file["sp_parameters"][batch_idx]
                return PyTorchFault(
                    selected_module_name,
                    module_idx,
                    torch_mask,
                    torch_values,
                    sp_class,
                    sp_param,
                )
            finally:
                member_file.close()


    def collate_fn(self, data: Sequence[PyTorchFaultBatch]):
        return fault_collate_fn(data)

    def __len__(self) -> int:
        return self.n_faults


class PyTorchLazyFaultList(IterableDataset[PyTorchFault]):
    def __init__(
        self,
        fault_list_generator: PyTorchFaultList,
        module: str,
        n_faults: int,
    ) -> None:
        super().__init__()
        self.fault_list_generator = fault_list_generator
        self.module = module
        self.n_faults = n_faults


    def __iter__(self):

        def convert_to_tensor(generated_fault : Tuple[str, int, PyTorchFaultBatch]):
            module_name, fault_id, fault = generated_fault

            return PyTorchFault(
                module_name,
                fault_id,
                torch.from_numpy(fault.corrupted_value_mask[0]),
                torch.from_numpy(fault.corrupted_values),
                fault.spatial_pattern_names[0],
                fault.sp_parameters[0]
            )

        return map(convert_to_tensor, self.fault_list_generator.module_fault_list_generator(
                    self.module, self.n_faults, 1
        ))
    
    def collate_fn(self, data: Sequence[PyTorchFaultBatch]):
        return fault_collate_fn(data)
