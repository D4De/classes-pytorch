import os
import json
import numpy as np
import pandas as pd
import torch.nn as nn

from typing import Callable, Mapping, Sequence

from classes.pattern_generators import PatternGenerator, get_default_generators
from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault_generator import FaultGenerator
from classes.simulators.pytorch.error_model_merger import merge_error_models


hyperparameters = [
    'Channels_in',
    'Channels_out',
    'Input_size',
    'Kernel_size',
    'Padding',
]

non_SDC_fields = [
    'Masked',
    'Crash+Hang',
]

def load_error_models(model_folder_path: str, model_df_filepath: str):
    """
    Loads the error model json files and the corresponding dataframe with the models' parameters.
    The hyperparameters are also normalized.
    
    Returns the dataframe and the dictionary of error model dictionaries.
    """

    # look for the model dataframe file
    if not os.path.exists(model_df_filepath):
        raise FileNotFoundError(f'Error models dataframe file {model_df_filepath} does not exist')
    
    model_df_ext = os.path.splitext(model_df_filepath)[1]
    match model_df_ext:
        case '.csv': model_parameters = pd.read_csv(model_df_filepath, index_col='Layer')
        case '.xlsx': model_parameters = pd.read_excel(model_df_filepath, sheet_name=0, index_col='Layer')
        case _ : raise ValueError(f'Error models dataframe file is of unsupported type {model_df_ext}')

    # normalize hyperparameters

    model_hyperparameter_cols = model_parameters[hyperparameters]
    h_min = model_hyperparameter_cols.min(axis=1)
    h_max = model_hyperparameter_cols.max(axis=1)
    model_parameters[hyperparameters] = model_hyperparameter_cols.sub(h_min, axis=0).divide(h_max-h_min, axis=0)

    # look for the error models and create the fault generators from them
    model_folder_json_files = list(filter(
        lambda x: x.endswith('.json'),
        os.listdir(model_folder_path)
    ))

    error_model_dicts = {}
    for file_name in model_folder_json_files:
        operator_name = file_name[:-5]
        file_path = os.path.join(model_folder_path, file_name)

        # load and save json dict for error model
        with open(file_path) as f:
            error_model_dict = json.load(f)
            error_model_dicts[operator_name] = error_model_dict

    return model_parameters, error_model_dicts


def interpolation_nearest_neighbors_L2(
    layer_hyperparameters: pd.Series,
    error_models_df: pd.DataFrame,
    error_model_dicts: list[dict],
    num_neighbors: int = 5,
):
    """
    Selects the 'num_neighbors' closest models to the given one using the L2 distance and the normalized hyperparameters.
    Builds a new error model and assigns it a name. Returns the new name, the new model and the interpolated frequencies for it.
    """
    new_model_num = len(error_models_df) + 1
    new_model_name = f'newmerge_{new_model_num}'

    model_hyperparameter_cols = error_models_df[hyperparameters]

    # compute parameter distances for all models and take the smallest ones
    distances = ((model_hyperparameter_cols - layer_hyperparameters) ** 2).sum(axis=1)
    min_distance_indices = distances.argsort()[:num_neighbors]

    closest_model_rows = error_models_df.iloc[min_distance_indices]
    closest_model_names = closest_model_rows.index.tolist()

    best_model_dictionaries = [error_model_dicts[name] for name in closest_model_names]

    # interpolate and add new model
    new_model_dict = merge_error_models(best_model_dictionaries)
    error_model_dicts[new_model_name] = new_model_dict

    # build dataframe row for the new model
    # compute average frequencies for non-hyperparameter columns
    new_row_frequencies = closest_model_rows.drop(hyperparameters, axis=1).mean()
    # concatenate with new hyperparameters
    new_row = pd.concat([
        layer_hyperparameters,
        new_row_frequencies
    ])
    error_models_df.loc[new_model_name] = new_row

    return new_model_name, new_row_frequencies


def map_layers_to_error_models(
    network: nn.Module,
    layer_names: list[str],
    layer_shapes: dict,
    error_models_df: pd.DataFrame,
    error_model_dicts: dict[str, dict],
    logger,
    generator_mapping: Mapping[str, PatternGenerator] = get_default_generators(),
    layout="CHW",
    interpolation_fn=interpolation_nearest_neighbors_L2,
) -> tuple[pd.DataFrame, dict[str, FaultGenerator]]:
    """
    Uses the hyperparameters of each injectable layer to find a matching error model (either an existing one or by interpolating
    existing ones). At the same time, the SDC frequency for each layer is determined via its error model and stored in a dictionary.
    The selected error models are then used to build a set of fault generators.

    The first return value is a DataFrame mapping each layer to its SDC and spatial class frequencies.
    The second return value is a dictionary mapping each layer to its fault generator.
    """
    sdc_frequencies = pd.DataFrame(columns=error_models_df.columns) # layer name to frequencies
    sdc_frequencies.drop(columns=hyperparameters + non_SDC_fields, inplace=True) # remove unneccessary columns
    
    fault_generators: dict[FaultGenerator] = {} # error model name to fault generator
    layer_to_fault_generator: dict[FaultGenerator] = {}

    for layer_name in layer_names:
        layer: nn.Conv2d = network.get_submodule(layer_name)
        layer_input_size: int = layer_shapes[layer_name][1][-1]

        layer_hyperparameters = pd.Series({
            'Channels_in' : layer.in_channels,
            'Channels_out': layer.out_channels,
            'Input_size'  : layer_input_size,
            'Kernel_size' : layer.kernel_size[-1],
            'Padding'     : layer.padding[-1],
        })
        logger.info(f'Looking for best error model for network layer with hyperparameters {layer_hyperparameters.values.tolist()}.')

        # normalize module parameters
        min_hyper: float = layer_hyperparameters.min()
        max_hyper: float = layer_hyperparameters.max()
        layer_hyperparameters: pd.Series = (layer_hyperparameters - min_hyper) / (max_hyper - min_hyper)

        # check the available models: if a matching one is found, record the SDC frequency and build the fault generator
        for model_name, model_hyper in error_models_df[hyperparameters].iterrows():
            if np.allclose(layer_hyperparameters, model_hyper):
                logger.info(f'Found exact match for error model: {model_name}.')

                model_df_row: pd.Series = error_models_df.loc[model_name].copy() # copy the model row
                model_df_row.drop(hyperparameters + non_SDC_fields, inplace=True) # drop unnecessary columns
                sdc_frequencies.loc[layer_name] = model_df_row # add new row to the dataframe
                
                if model_name not in fault_generators:
                    # fault generator not yet built: create it
                    error_model = ErrorModel.from_json_dict(error_model_dicts[model_name])
                    fault_generators[model_name] = FaultGenerator(
                        error_model,
                        generator_mapping=generator_mapping,
                        layout=layout,
                    )

                layer_to_fault_generator[layer_name] = fault_generators[model_name]
                break

        # no matching model: create a new one via interpolation
        else:
            logger.info('No matching error model available. Proceeding with interpolation.')
            new_model_name, new_model_frequencies = interpolation_fn(layer_hyperparameters, error_models_df, error_model_dicts)

            # record SDC frequency
            sdc_frequencies.loc[layer_name] = new_model_frequencies

            # build new fault generator
            error_model = ErrorModel.from_json_dict(error_model_dicts[new_model_name])
            fault_generators[new_model_name] = FaultGenerator(
                error_model,
                generator_mapping=generator_mapping,
                layout=layout,
            )

            # associate fault generator to layer
            layer_to_fault_generator[layer_name] = fault_generators[new_model_name]

            logger.info(f'Created and saved new error model {new_model_name}.')


    return sdc_frequencies, layer_to_fault_generator




# DEPRECATED
def create_module_to_generator_mapper_dynamic(
    model_folder_path: str,
    model_df_filepath: str,
    logger,
    generator_mapping: Mapping[str, PatternGenerator] = get_default_generators(),
    layout="CHW",
    interpolation_fn=interpolation_nearest_neighbors_L2,
) -> Callable[[nn.Module, Sequence[int]], tuple[FaultGenerator, float] | None]:
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

    # look for the model dataframe file
    if not os.path.exists(model_df_filepath):
        raise FileNotFoundError(f'Error models dataframe file {model_df_filepath} does not exist')
    
    logger.info(f'Loading error models dataframe file {model_df_filepath}.')
    model_df_ext = os.path.splitext(model_df_filepath)[1]
    match model_df_ext:
        case '.csv': model_parameters = pd.read_csv(model_df_filepath, index_col='Layer')
        case '.xlsx': model_parameters = pd.read_excel(model_df_filepath, sheet_name=0, index_col='Layer')
        case _ : raise ValueError(f'Error models dataframe file is of unsupported type {model_df_ext}')

    # extract SDC column
    SDC_col = model_parameters['SDC']

    # normalize hyperparameters
    hyperparameters = [
        'Channels_in',
        'Channels_out',
        'Input_size',
        'Kernel_size',
        'Padding',
    ]
    model_hyperparameter_cols = model_parameters[hyperparameters]
    h_min = model_hyperparameter_cols.min(axis=1)
    h_max = model_hyperparameter_cols.max(axis=1)
    model_hyperparameter_cols = model_hyperparameter_cols.sub(h_min, axis=0).divide(h_max-h_min, axis=0)

    # look for the error models and create the fault generators from them
    model_folder_json_files = list(filter(
        lambda x: x.endswith('.json'),
        os.listdir(model_folder_path)
    ))

    error_model_dicts = {}
    fault_generators = {}
    for file_name in model_folder_json_files:
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
    

    def _mapper(module: nn.Module, input_shape) -> tuple[FaultGenerator, float] | None:
        """
        Returns the FaultGenerator matching the provided module. If the module fits one of the available error models, its
        generator is returned directly. If none of the models match, they are interpolated to produce a suitable generator.
        Along with the FaultGenerator, the SDC frequency of the selected (or interpolated) error model is returned so that
        it may later be used to rescale the simulation results.
        Returns None if the module is of an unsupported type.
        """
        if not isinstance(module, nn.Conv2d):
            # not a convolution: skip
            return None

        # for now, assume the input shape to always be square
        module_hyperparameters = pd.Series({
            'Channels_in':  module.in_channels,
            'Channels_out': module.out_channels,
            'Input_size': input_shape[-1],
            'Kernel_size': module.kernel_size[-1],
            'Padding': module.padding[-1],
        })
        logger.info(f'Looking for best error model for network layer with hyperparameters {module_hyperparameters.values.tolist()}.')

        # normalize module parameters
        min_hyper: float = module_hyperparameters.min()
        max_hyper: float = module_hyperparameters.max()
        module_hyperparameters: pd.Series = (module_hyperparameters - min_hyper) / (max_hyper - min_hyper)

        # check the available models: if a matching one is found, return its generator and SDC frequency
        for model_name, model_hyper in model_hyperparameter_cols.iterrows():
            if np.allclose(module_hyperparameters, model_hyper):
                logger.info('Found exact match for error model.')
                return fault_generators[model_name], float(SDC_col[model_name])

        # no matching model available: interpolate and add the new one to the list
        logger.info('No matching error model available. Proceeding with interpolation.')
        new_model_name, new_model_dict, new_model_sdc = interpolation_fn(
            module_hyperparameters,
            model_hyperparameter_cols,
            SDC_col,
            error_model_dicts
        )

        # add new parameters and frequency
        model_hyperparameter_cols.loc[new_model_name] = module_hyperparameters
        SDC_col.loc[new_model_name] = pd.Series({'SDC': new_model_sdc})

        # add new model dictionary to available ones
        error_model_dicts[new_model_name] = new_model_dict

        # build new fault generator
        new_error_model = ErrorModel.from_json_dict(new_model_dict)
        fault_generators[new_model_name] = FaultGenerator(
            new_error_model, generator_mapping=generator_mapping, layout=layout
        )

        logger.info(f'Created and saved new error model {new_model_name}.')
        return fault_generators[new_model_name], new_model_sdc

    return _mapper