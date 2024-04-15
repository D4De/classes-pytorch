from typing import Callable
import torch
import torch.nn as nn

from classes.simulators.pytorch.pytorch_fault import PyTorchFault, PyTorchFaultBatch
from contextlib import contextmanager


def create_simulator_hook(pytorch_fault: PyTorchFault):
    """
    Creates a pytorch forward hook that can be attached to
    """

    def _error_simulator_hook(module, input, output):
        mask = pytorch_fault.corrupted_value_mask
        output[:, mask != 0] = pytorch_fault.corrupted_values

        return output

    return _error_simulator_hook


def create_batched_simulator_hook(pytorch_fault: PyTorchFaultBatch):
    def _error_simulator_hook(module, input, output):
        mask = pytorch_fault.corrupted_value_mask
        output[mask != 0] = pytorch_fault.corrupted_values

        return output

    return _error_simulator_hook


@contextmanager
def applied_hook(module: nn.Module, hook: Callable):
    try:
        handle = module.register_forward_hook(hook)
        yield
    finally:
        handle.remove()
