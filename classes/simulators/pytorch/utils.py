from typing import Any, Generator
import torch
import torch.nn as nn


def traverse_network(network : nn.Module) -> Generator[nn.Module, Any, None]:
    for name, module in network.named_modules():
        if isinstance(module, nn.Sequential)