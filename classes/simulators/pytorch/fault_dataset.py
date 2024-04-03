from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import zipfile
import os
import json
import shutil
import numpy as np

class FaultListFromZipFile(Dataset):
    def __init__(self, fault_list_path : str, module : str, include_sp_parameters = False) -> None:
        super().__init__()
        self.fault_list_path = fault_list_path
        self.module = module
        self.include_sp_parameters = include_sp_parameters
        self.fault_list_info = None
        self.tmp_path = None

        with zipfile.ZipFile(self.fault_list_path, 'r') as zipf:
            with zipf.open('fault_list.json') as f:
                fault_list_info = json.load(f)
                self.input_shape = fault_list_info['input_shape']
                self.batch_dimension = fault_list_info['batch_dimension']
                self.modules_output_shapes = fault_list_info['modules_output_shapes']
                self.n_faults_per_module = fault_list_info['n_faults_per_module']
                self.fault_batch_size = fault_list_info['fault_batch_size']
                self.n_injectable_layers = fault_list_info['n_injectable_layers']
        

        self.n_faults = self.n_faults_per_module

    def extract_fault_list(self):
        if self.is_extracted():
            return
        if self.fault_list_path.endswith('.zip'):
            self.tmp_path = os.path.join(self.fault_list_path[:-4] + '_tmp')
        else:
            self.tmp_path = os.path.join(self.fault_list_path + '_tmp')
        with zipfile.ZipFile(self.fault_list_path, 'r') as zipf:
            os.makedirs(self.tmp_path, exist_ok=True)
            print(f'Extracted faultlist to {self.tmp_path}')
            zipf.extractall(path=self.tmp_path)

    def delete_extracted_fault_list(self):
        if self.is_extracted():
            assert self.tmp_path is not None
            shutil.rmtree(self.tmp_path, ignore_errors=True)
            self.tmp_path = None
        

    def is_extracted(self) -> bool:
        return self.tmp_path is not None and os.path.exists(self.tmp_path)

    __enter__ = extract_fault_list

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete_extracted_fault_list()

    def __getitem__(self, index):
        if not self.is_extracted():
            raise FileNotFoundError(f"""
                        Fault list zipfile is not extracted and can not be used.
                        Before accessing the dataset you should extract the file calling 
                        
                        Please call the method extract_fault_list before accessing the 
                        dataset, or wrap everything inside a 'with' statement""")
        assert self.tmp_path is not None

        module_fault_path = os.path.join(self.tmp_path, self.module)
        file_idx = index // self.fault_batch_size
        batch_idx = index % self.fault_batch_size
        file_path = os.path.join(module_fault_path, f"{file_idx}_faults_{self.module}.npz")
        
        fault_batch = np.load(file_path, allow_pickle=True)
        mask = fault_batch["masks"][batch_idx]
        slice_begin, slice_end = fault_batch["values_index"][batch_idx:batch_idx+2]
        #print(fault_batch["values_index"])
        values = fault_batch["values"][slice_begin:slice_end]

        torch_mask = torch.from_numpy(mask)
        torch_values = torch.from_numpy(values)
        length = torch_values.numel()

        if self.include_sp_parameters:
            sp_class = fault_batch["sp_classes"][batch_idx]
            sp_param = fault_batch["sp_parameters"][batch_idx]
            return torch_mask, torch_values, torch.LongTensor([0, length]), sp_class, sp_param
        else:
            return torch_mask, torch_values, torch.LongTensor([0, length])
    
    def collate_fn(self, data):
        masks, values, values_idxs = zip(*data)

        torch_values_idxs = torch.stack(values_idxs)

        batch_masks = torch.stack(masks, dim=0)
        batch_values = torch.concat(values)
        batch_values_idxs = torch.zeros(len(masks) + 1)
        batch_values_idxs[1:] = torch.cumsum(torch_values_idxs[:,1], dim=0)

        return batch_masks, batch_values, batch_values_idxs
    

    def __len__(self) -> int:
        return self.n_faults