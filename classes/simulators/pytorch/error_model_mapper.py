import torch
import torch.nn as nn

import os

from typing import Callable, Dict, Literal, Mapping, Optional


from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.pattern_generators import PatternGenerator, get_default_generators


ModuleToFaultGeneratorMapper = Callable[[str, nn.Module], Optional[FaultGenerator]]


def create_module_to_generator_mapper(
    model_folder_path="error_models/models",
    conv_strategy: Literal["conv_gemm", "conv_fft", "conv_win"] = "conv_gemm",
    generator_mapping: Mapping[str, PatternGenerator] = get_default_generators(),
    layout="CHW",
) -> ModuleToFaultGeneratorMapper:
    """
    Creates a function that maps each Module of a PyTorch network to a FaultGenerator
    configure with the relevant model error for that Module.

    This function returns a default implementation of the mapper that consides only the operator
    of nn.Module and returns a pre-created instance of `FaultGenerator` loaded with the
    error models of that operator.


    Args
    ----
    * ``model_folder_path : str``. The path to a folder containing all the .json error models files
    * ``conv_strategy :  Literal["conv_gemm", "conv_fft", "conv_win"]``. The convolution strategy to choose for the ``nn.Conv2d`` layers.
        Each convolution strategy has its own different error models
    *`` generator_mapping : Dict[str, PatternGenerator]``: A dictionary that maps the names of the spatial classes with their genetor functions.
        If not specified, the default generator returned by the function `classes.pattern_generators.get_default_generators()` is used.
    * ``layout : Literal["CHW", "HWC"]. The axis ordering used for generating the faults. "CHW" is the default one for pytorch.

    Returns
    ----
    A function that takes in input the name of a Module (str) and the ``nn.Module`` itself and
    returns the an instance of a `FaultGenerator` that generate the faults according
    to the correct error model.

    When specifying a custom fault_generator is advisable, if possible, to not generate a new `FaultGenerator` each time the returned function is invoked, but
    instead it is better to instatiate the `FaultGenerator` in the closure of the function generated and
    return a reference to the pre-constructed object.

    ```
    def make_custom_fault_generator_mapper(...):
        ...
        # AVOID (if possible)
        def _custom_mapper(name, module):
            if name == 'foo':
                return FaultGenerator(...)
            return None
        return _custom_mapper
        ...
        # YES
        foo_generator = FaultGenerator(...)
        def _custom_mapper(name, module):
            if name == 'foo':
                return foo_generator
            return None
        return _custom_mapper
    ```
    """

    error_model_file_names = os.listdir(model_folder_path)
    fault_generators = {}
    for file_name in error_model_file_names:
        file_path = os.path.join(model_folder_path, file_name)
        error_model = ErrorModel.from_json_file(file_path)
        operator_name = file_name[:-5]
        fault_generators[operator_name] = FaultGenerator(
            error_model, generator_mapping=generator_mapping, layout=layout
        )
    # mapping between nn.Module subtypes and the name of the error models
    # convolution has multiple error models, the one chosen is passed as argument
    module_to_err_model_mapping: dict[type, str] = {
        nn.Conv2d: conv_strategy,
        nn.MaxPool2d: "maxpool",
        nn.AvgPool2d: "avgpool",
        nn.BatchNorm2d: "batchnorm",
        #nn.ReLU: "relu",
        #nn.Sigmoid: "sigmoid",
        #nn.Tanh: "tanh",
        #nn.ELU: "elu",
    }

    def _mapper(module_name: str, module: nn.Module):
        for module_type, error_model_key in module_to_err_model_mapping.items():
            if isinstance(module, module_type):
                return fault_generators[error_model_key]

        return None

    return _mapper
