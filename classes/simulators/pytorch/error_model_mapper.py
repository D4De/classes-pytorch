import torch
import torch.nn as nn

import os

from typing import Callable, Dict, Literal, Optional


from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.pattern_generators import PatternGenerator, get_default_generators


ModuleToFaultGeneratorMapper = Callable[[str, nn.Module], Optional[FaultGenerator]]


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
    
    module_to_err_model_mapping : dict[type, str] = {
        nn.Conv2d: conv_strategy,
        nn.MaxPool2d: 'maxpool',
        nn.AvgPool2d: 'avgpool',
        nn.BatchNorm2d: 'batchnorm',
        nn.ReLU: 'relu',
        nn.Sigmoid: 'sigmoid',
        nn.Tanh: 'tanh',
        nn.ELU: 'elu'
    }

    def _mapper(module_name: str, module: nn.Module):
        for module_type, error_model_key in module_to_err_model_mapping.items():
            if isinstance(module, module_type):
                return fault_generators[error_model_key]

        return None

    return _mapper
