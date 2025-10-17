import yaml
import torch.nn as nn

import os
import numpy as np

from random import choice
from typing import Callable, Dict, Literal, Mapping, Optional

from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.pattern_generators import PatternGenerator, get_default_generators


ModuleToFaultGeneratorMapper = Callable[[str, nn.Module], Optional[FaultGenerator]]


def create_module_to_generator_mapper(
    model_folder_path="error_models/faulty_thread_models",
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
        #nn.MaxPool2d: "maxpool",
        #nn.AvgPool2d: "avgpool",
        #nn.BatchNorm2d: "batchnorm",
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


# --------------------------------------------------------------------------------------------------

def interpolation_nearest_neighbor_L2(
        module_parameters: list,
        all_model_parameters: dict[str, list],
        fault_generators,
        logger,
    ):
    """
    Selects the error model closest to the module by computing the L2 distance between hyperparameter lists and taking the minimum.
    """
    module_param_np = np.array(module_parameters)

    min_distance = np.inf
    best_model: str = None
    exact_matches: list[str] = []

    for model_name, parameters in all_model_parameters.items():
        param_np = np.array(parameters)
        distance_squared = np.sum(np.pow(module_param_np - param_np, 2))

        if distance_squared == 0.0:
            # found model that matches exactly
            exact_matches.append(model_name)
            min_distance = 0.0
            continue

        if distance_squared < min_distance:
            # found new best model
            min_distance = distance_squared
            best_model = model_name

    if exact_matches:
        # some models that match the layer exactly are available: choose randomly
        best_model = choice(exact_matches)

    logger.info(f'Best error model is {best_model}.')
    return fault_generators[best_model]


def create_module_to_generator_mapper_dynamic(
    model_folder_path: str,
    logger,
    model_parameters_filename='model_parameters.yaml',
    generator_mapping: Mapping[str, PatternGenerator] = get_default_generators(),
    layout="CHW",
    interpolation_fn=interpolation_nearest_neighbor_L2,
) -> ModuleToFaultGeneratorMapper:
    """
    As the default generator mapper function, this builds a function mapping a PyTorch module to a FaultGenerator.
    The difference is that this function looks for a more precise match between a module and an error model: it uses the whole
    module (specifically, its hyperparameters) and looks for a model obtained from a module with the same ones. If it can't find
    a match, it interpolates existing models.

    Note that this more complex matching procedure can only be performed after an initial network profiling to obtain the module's input
    shape. Furthermore, the error model folder must contain a 'model_parameters.yaml' file mapping error model names to their
    respective hyperparameters.
    """
    logger.info('Building module->fault generator mapping function...')

    # look for the model parameters file
    model_parameters_filepath = os.path.join(model_folder_path, model_parameters_filename)
    if not os.path.exists(model_parameters_filepath):
        raise FileNotFoundError(f'Model parameters file {model_parameters_filename} not found in model folder directory {model_folder_path}')
    
    logger.info(f'Loading model parameters yaml file {model_parameters_filepath}.')
    with open(model_parameters_filepath) as f:
        model_parameters = yaml.load(f, yaml.SafeLoader)

    # look for the error models and create the fault generators from them
    model_folder_filenames = os.listdir(model_folder_path)
    model_folder_filenames.remove(model_parameters_filename)

    fault_generators = {}
    for file_name in model_folder_filenames:
        file_path = os.path.join(model_folder_path, file_name)
        error_model = ErrorModel.from_json_file(file_path)
        operator_name = file_name[:-5]
        fault_generators[operator_name] = FaultGenerator(
            error_model, generator_mapping=generator_mapping, layout=layout
        )
        logger.info(f'Error model {operator_name} loaded.')
    

    def _mapper(module: nn.Module, input_shape):
        if not isinstance(module, nn.Conv2d):
            # not a convolution: skip
            return None

        # for now, assume the input shape to always be square
        module_hyperparameters = [
            module.in_channels,
            module.out_channels,
            input_shape[-1],
            module.kernel_size[-1],
            module.padding[-1]
        ]
        logger.info(f'Looking for best error model for network layer with hyperparameters {module_hyperparameters}.')

        # look for error model that matches exactly or take the closest one
        return interpolation_fn(module_hyperparameters, model_parameters, fault_generators, logger)

    return _mapper