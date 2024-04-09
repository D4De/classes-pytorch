from dataclasses import dataclass, astuple
from typing import Any, Dict, Sequence

import torch


@dataclass
class PyTorchFault:
    fault_id : int
    corrupted_value_mask : torch.Tensor
    corrupted_values : torch.Tensor
    spatial_pattern_name : str
    sp_parameters : Dict[str, Any]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))
    
    def to(self, device=None, dtype=None, non_blocking=False, copy=False):
        """
        Moves the `torch.Tensor` objects that constitute the `PyTorchFaultBatch` to another device.
        Has the same semantics of PyTorch's `Tensor.to` method
        """
        self.corrupted_value_mask = self.corrupted_value_mask.to(device, dtype, non_blocking, copy) 
        self.corrupted_values = self.corrupted_values.to(device, dtype, non_blocking, copy) 

@dataclass
class PyTorchFaultBatch:
    fault_ids : Sequence[int]
    corrupted_value_mask : torch.Tensor
    corrupted_values : torch.Tensor
    corrupted_values_index : torch.LongTensor
    spatial_pattern_names : Sequence[str]
    sp_parameters : Sequence[Dict[str, Any]]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))
    
    def batch_size(self) -> int:
        return self.corrupted_value_mask.size(0)
    
    def get_element(self, idx):
        value_begin, value_end = self.corrupted_values_index[idx:idx+1]
        return PyTorchFault(
            self.fault_ids[idx],
            self.corrupted_value_mask[idx],
            self.corrupted_values[value_begin:value_end],
            self.spatial_pattern_names[idx], #type: ignore
            self.sp_parameters[idx]
        )
    
    def to(self, device=None, dtype=None, non_blocking=False, copy=False):
        """
        Moves the `torch.Tensor` objects that constitute the `PyTorchFaultBatch` to another device.
        Has the same semantics of PyTorch's `Tensor.to` method
        """
        self.corrupted_value_mask = self.corrupted_value_mask.to(device, dtype, non_blocking, copy) 
        self.corrupted_values = self.corrupted_values.to(device, dtype, non_blocking, copy) 
        self.corrupted_values_index = self.corrupted_values_index.to(device, dtype, non_blocking, copy) # type: ignore
