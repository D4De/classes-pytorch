import torch
import torch.nn as nn

import os

from typing import Dict, Literal


from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.pattern_generators import PatternGenerator, get_default_generators
from classes.simulators.pytorch.fault_list_generator import ModuleToFaultGeneratorMapper


def create_module_to_generator_mapper(
    model_folder_path="error_models/models",
    conv_strategy: Literal["conv_gemm", "conv_fft", "conv_win"] = "conv_gemm",
    generator_mapping: Dict[str, PatternGenerator] = get_default_generators(),
    layout="CHW",
) -> ModuleToFaultGeneratorMapper:

    error_model_file_names = os.listdir(model_folder_path)
    fault_generators = {}
    for file_name in error_model_file_names:
        file_path = os.path.join(model_folder_path, file_name)
        error_model = ErrorModel.from_json_file(file_path)
        operator_name = file_name[:-5]
        fault_generators[operator_name] = FaultGenerator(
            error_model, generator_mapping=generator_mapping, layout=layout
        )

    def _mapper(module_name: str, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            return fault_generators[conv_strategy]
        if isinstance(module, nn.MaxPool2d):
            return fault_generators["maxpool"]
        if isinstance(module, nn.AvgPool2d):
            return fault_generators["avgpool"]
        if isinstance(module, nn.BatchNorm2d):
            return fault_generators["batchnorm"]
        if isinstance(module, nn.ReLU):
            return fault_generators["relu"]
        if isinstance(module, nn.Sigmoid):
            return fault_generators["sigmoid"]
        if isinstance(module, nn.Tanh):
            return fault_generators["tahn"]
        if isinstance(module, nn.ELU):
            return fault_generators["elu"]
        return None

    return _mapper
