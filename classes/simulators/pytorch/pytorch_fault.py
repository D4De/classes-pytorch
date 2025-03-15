from dataclasses import dataclass, astuple
from typing import Any, Dict, List, Optional, Sequence

import torch

from classes.fault_generator.fault import Fault, FaultBatch


@dataclass
class PyTorchFault:
    module_name: str
    fault_id: int
    corrupted_value_mask: torch.Tensor
    corrupted_values: torch.Tensor
    spatial_pattern_name: str
    sp_parameters: Dict[str, Any]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))
    

    @classmethod
    def from_fault(cls, fault : Fault, module_name : str = '', fault_id = 0):

        return cls(
            module_name,
            fault_id,
            torch.from_numpy(fault.corrupted_value_mask),
            torch.from_numpy(fault.corrupted_values),
            fault.spatial_pattern_name,
            fault.sp_parameters
        )

    def to(self, device=None, dtype=None, non_blocking=False, copy=False):
        """
        Moves the `torch.Tensor` objects that constitute the `PyTorchFaultBatch` to another device.
        Has the same semantics of PyTorch's `Tensor.to` method
        """
        self.corrupted_value_mask = self.corrupted_value_mask.to(
            device, dtype, non_blocking, copy
        )
        self.corrupted_values = self.corrupted_values.to(
            device, dtype, non_blocking, copy
        )


@dataclass
class PyTorchFaultBatch:
    module_names: Sequence[str]
    fault_ids: Sequence[int]
    corrupted_value_mask: torch.Tensor
    corrupted_values: torch.Tensor
    corrupted_values_index: torch.LongTensor
    spatial_pattern_names: Sequence[str]
    sp_parameters: Sequence[Dict[str, Any]]

    def __iter__(self):
        # Enable tuple unpacking
        return iter(astuple(self))
    
    @classmethod
    def from_fault_batch(cls, fault_batch : FaultBatch, module_name : str = '', fault_indexes: Optional[List[int]] = None):
        if fault_indexes is None:
            fault_indexes = list(range(len(fault_batch)))
        
        return cls(
            [module_name for i in range(len(fault_batch))],
            fault_indexes,
            torch.from_numpy(fault_batch.corrupted_value_mask),
            torch.from_numpy(fault_batch.corrupted_values),
            torch.from_numpy(fault_batch.corrupted_values_index).long(),
            fault_batch.spatial_pattern_names,
            fault_batch.sp_parameters
        )
        
    def batch_size(self) -> int:
        return self.corrupted_value_mask.size(0)

    def get_element(self, idx: int):
        value_begin, value_end = self.corrupted_values_index[idx : idx + 1]
        return PyTorchFault(
            self.module_names[idx],
            self.fault_ids[idx],
            self.corrupted_value_mask[idx],
            self.corrupted_values[value_begin:value_end],
            self.spatial_pattern_names[idx],  # type: ignore
            self.sp_parameters[idx],
        )

    def to(self, device=None, dtype=None, non_blocking=False, copy=False):
        """
        Moves the `torch.Tensor` objects that constitute the `PyTorchFaultBatch` to another device.
        Has the same semantics of PyTorch's `Tensor.to` method
        """
        self.corrupted_value_mask = self.corrupted_value_mask.to(
            device, dtype, non_blocking, copy
        )
        self.corrupted_values = self.corrupted_values.to(
            device, dtype, non_blocking, copy
        )
        self.corrupted_values_index = self.corrupted_values_index.to(device, dtype, non_blocking, copy)  # type: ignore
