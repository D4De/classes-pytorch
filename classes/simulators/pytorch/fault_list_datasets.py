from dataclasses import dataclass
from typing import Optional, Sequence

from classes.simulators.pytorch.fault_list import PyTorchFaultList, PyTorchFaultListInfo
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

import tarfile
import os
import json
import shutil
import numpy as np

from classes.simulators.pytorch.pytorch_fault import PyTorchFault, PyTorchFaultBatch


class FaultListFromTarFile(Dataset[PyTorchFault]):
    def __init__(self, fault_list_path: str, module: Optional[str] = None) -> None:
        super().__init__()
        self.fault_list_path = fault_list_path
        self.module = module
        self.include_sp_parameters = True
        self.fault_list_info = None
        self.tmp_path = None

        self.info = PyTorchFaultListInfo.load_fault_list_info(self.fault_list_path)

        self.n_faults = self.info.n_faults_per_module

    def __getitem__(self, index: int) -> PyTorchFault:

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
                    index,
                    torch_mask,
                    torch_values,
                    sp_class,
                    sp_param,
                )
            finally:
                member_file.close()

    def collate_fn(self, data: Sequence[PyTorchFaultBatch]):
        print(len(data))
        names, fault_ids, masks, values, sp_classes, sp_parameters = zip(*data)
        sp_classes = [cl[0] for cl in sp_classes]
        sp_parameters = [prm[0] for prm in sp_parameters]

        torch_values_idxs = torch.LongTensor(
            [value_tensor.size() for value_tensor in values]
        )

        batch_masks = torch.concat(masks, dim=0)
        batch_values = torch.concat(values)

        batch_values_idxs = torch.LongTensor(
            torch.zeros(len(masks) + 1).long()
        )  # sono veramente long
        batch_values_idxs[1:] = torch.cumsum(torch_values_idxs, dim=0)

        return PyTorchFaultBatch(
            names,
            fault_ids,
            batch_masks,
            batch_values,
            batch_values_idxs,
            sp_classes,
            sp_parameters,
        )

    def __len__(self) -> int:
        return self.n_faults


class FaultListFromGenerator(IterableDataset):
    def __init__(
        self,
        fault_list_generator: PyTorchFaultList,
        module: str,
        n_faults: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.fault_list_generator = fault_list_generator
        self.module = module
        self.n_faults = n_faults
        self.batch_size = batch_size

    def __iter__(self):
        return self.fault_list_generator.module_fault_list_generator(
            self.module, self.n_faults, self.batch_size
        )
