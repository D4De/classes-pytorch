from typing import Any, Optional, Sequence

from classes.simulators.pytorch.fault_list_generator import PyTorchFaultListGenerator
import torch
from torch.utils.data import Dataset

import tarfile
import os
import json
import numpy as np

from classes.simulators.pytorch.pytorch_fault import PyTorchFault, PyTorchFaultBatch

class FaultListFromTarFile(Dataset[PyTorchFault]):
    def __init__(self, fault_list_path : str, module : str) -> None:
        super().__init__()
        self.fault_list_path = fault_list_path
        self.module = module
        self.include_sp_parameters = True
        self.fault_list_info = None
        self.tmp_path = None

        with tarfile.TarFile(self.fault_list_path, 'r') as tarf:
            member_file = tarf.extractfile('fault_list.json')
            if not member_file:
                raise FileNotFoundError(f'Archive member file fault_list.json not found')
            try:
                fault_list_info = json.load(member_file)
                self.input_shape = fault_list_info['input_shape']
                self.batch_dimension = fault_list_info['batch_dimension']
                self.modules_output_shapes = fault_list_info['modules_output_shapes']
                self.n_faults_per_module = fault_list_info['n_faults_per_module']
                self.fault_batch_size = fault_list_info['fault_batch_size']
                self.n_injectable_layers = fault_list_info['n_injectable_layers']
            finally:
                member_file.close()

        self.n_faults = self.n_faults_per_module

    def __getitem__(self, index):

        module_fault_path = self.module
        file_idx = index // self.fault_batch_size
        batch_idx = index % self.fault_batch_size
        file_path = os.path.join(module_fault_path, f"{file_idx}_faults_{self.module}.npz")

        with tarfile.TarFile(self.fault_list_path, 'r') as tarf:
            member_file = tarf.extractfile(file_path)
            if not member_file:
                raise FileNotFoundError(f'Archive member file {member_file} not found')
            try:
                fault_batch = np.load(member_file, allow_pickle=True)
                mask = fault_batch["masks"][batch_idx]
                slice_begin, slice_end = fault_batch["values_index"][batch_idx:batch_idx+2]
                #print(fault_batch["values_index"])
                values = fault_batch["values"][slice_begin:slice_end]

                torch_mask = torch.from_numpy(mask)
                torch_values = torch.from_numpy(values)

                sp_class = fault_batch["sp_classes"][batch_idx]
                sp_param = fault_batch["sp_parameters"][batch_idx]
                return PyTorchFault(index, torch_mask, torch_values, sp_class, sp_param)
            finally:
                member_file.close()


    
    def collate_fn(self, data : Sequence[PyTorchFault]):
        indexes, masks, values, sp_classes, sp_parameters = zip(*data)

        torch_values_idxs = torch.LongTensor([value.dim(0) for value in values])

        batch_masks = torch.concat(masks, dim=0)
        batch_values = torch.concat(values)
       
        batch_values_idxs = torch.LongTensor(torch.zeros(len(masks) + 1).long())
        batch_values_idxs[1:] = torch.cumsum(torch_values_idxs[:,1], dim=0)

        return PyTorchFaultBatch(indexes, batch_masks, batch_values, batch_values_idxs, sp_classes, sp_parameters)
    

    def __len__(self) -> int:
        return self.n_faults
    
class FaultListFromGenerator(Dataset):
    def __init__(self, fault_list_generator : PyTorchFaultListGenerator, module : str, n_faults : int, batch_size : int) -> None:
        super().__init__()
        self.fault_list_generator = fault_list_generator
        self.module = module
        self.n_faults = n_faults
        self.batch_size = batch_size

    def __iter__(self):
        return self.fault_list_generator.module_fault_list_generator(self.module, self.n_faults, self.batch_size)
