import yaml
import torch.nn as nn

import os
import json
import numpy as np

from typing import Callable, Dict, Literal, Mapping, Optional

from classes.pattern_generators import PatternGenerator, get_default_generators
from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.simulators.pytorch.error_model_merger import merge_error_models

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

def interpolation_nearest_neighbors_L2(
        module_parameters: np.ndarray,
        all_model_parameters: dict[str, np.ndarray],
        error_model_dicts: dict[dict],
        num_neighbors: int = 5,
    ):
    """
    Selects the 'num_neighbors' closest models to the given one using the L2 distance and the normalized hyperparameters.
    Builds a new error model and assigns it a name.
    """
    new_model_num = len(all_model_parameters) + 1
    new_model_name = f'newmerge_{new_model_num}'

    distances = []
    # compute parameter distances for all models
    for parameters in all_model_parameters.values():
        distance_squared = np.sum(np.pow(module_parameters - parameters, 2))
        distances.append(distance_squared)

    # find the indices of the smallest distances
    smallest_indices = np.argsort(np.array(distances))[:num_neighbors]
    # collect the corresponding error model dictionaries
    best_model_names = [list(all_model_parameters.keys())[i] for i in smallest_indices]
    best_dictionaries = [error_model_dicts[name] for name in best_model_names]

    # interpolate
    new_model_dict = merge_error_models(best_dictionaries)

    return new_model_name, new_model_dict


def create_module_to_generator_mapper_dynamic(
    model_folder_path: str,
    logger,
    model_parameters_filename='model_parameters.yaml',
    generator_mapping: Mapping[str, PatternGenerator] = get_default_generators(),
    layout="CHW",
    interpolation_fn=interpolation_nearest_neighbors_L2,
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

    # normalize the model hyperparameters
    for modelname, hyper_set in model_parameters.items():
        hyper_set = np.array(hyper_set).astype(np.float64)
        max_hyper = np.max(hyper_set)
        min_hyper = np.min(hyper_set)
        normalized_set = (hyper_set - min_hyper) / (max_hyper - min_hyper)
        model_parameters[modelname] = normalized_set

    # look for the error models and create the fault generators from them
    model_folder_filenames = os.listdir(model_folder_path)
    model_folder_filenames.remove(model_parameters_filename)

    error_model_dicts = {}
    fault_generators = {}
    for file_name in model_folder_filenames:
        operator_name = file_name[:-5]
        file_path = os.path.join(model_folder_path, file_name)

        # load and save json dict for error model
        with open(file_path) as f:
            error_model_dict = json.load(f)
            error_model_dicts[operator_name] = error_model_dict

        error_model = ErrorModel.from_json_dict(error_model_dict, path=file_path)
        fault_generators[operator_name] = FaultGenerator(
            error_model, generator_mapping=generator_mapping, layout=layout
        )
        logger.info(f'Error model {operator_name} loaded.')
    

    def _mapper(module: nn.Module, input_shape):
        if not isinstance(module, nn.Conv2d):
            # not a convolution: skip
            return None

        # for now, assume the input shape to always be square
        module_hyperparameters = np.array([
            module.in_channels,
            module.out_channels,
            input_shape[-1],
            module.kernel_size[-1],
            module.padding[-1]
        ])
        logger.info(f'Looking for best error model for network layer with hyperparameters {module_hyperparameters}.')

        # normalize module parameters
        max_hyper = np.max(module_hyperparameters)
        min_hyper = np.min(module_hyperparameters)
        module_hyperparameters = (module_hyperparameters - min_hyper) / (max_hyper - min_hyper)

        # check the available models: if a matching one is found, return that
        for modelname, hyper_set in model_parameters.items():
            if np.allclose(module_hyperparameters, hyper_set):
                logger.info('Found exact match for error model.')
                return fault_generators[modelname]

        # no matching model available: interpolate and add the new one to the list
        logger.info('No matching error model available. Proceeding with interpolation.')
        new_model_name, new_model_dict = interpolation_fn(module_hyperparameters, model_parameters, error_model_dicts)

        # add new parameters
        model_parameters[new_model_name] = module_hyperparameters
        # add new model dictionary to available ones
        error_model_dicts[new_model_name] = new_model_dict
        # build new fault generator
        new_error_model = ErrorModel.from_json_dict(new_model_dict)
        fault_generators[new_model_name] = FaultGenerator(
            new_error_model, generator_mapping=generator_mapping, layout=layout
        )

        logger.info(f'Created and saved new error model {new_model_name}.')
        return fault_generators[new_model_name]

    return _mapper