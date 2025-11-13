import os
import yaml
import json
import pandas as pd

import error_models.injection_campaign_postprocessing.postprocessing_utils as utils

from argparse import ArgumentParser

def load_config_dict():
    parser = ArgumentParser()
    parser.add_argument('config_filepath', help='Path to the postprocess configuration yaml file.')
    args = parser.parse_args()

    config_path = os.path.realpath(args.config_filepath)
    if not os.path.exists(config_path):
        raise ValueError(f'Configuration file at {config_path} does not exist.')
    
    with open(config_path) as f:
        config_dict = yaml.load(f, yaml.SafeLoader)
    
    # check configuration consistency
    networks_and_layers: dict = config_dict['networks_and_layers']
    input_sizes: list = config_dict['network_input_sizes']
    if len(networks_and_layers) != len(input_sizes):
        raise ValueError(f'The dictionary of networks and the list of input sizes in the configuration file have different lengths.')

    return config_dict


def check_layers_and_adjust_models(config_dict: dict, initial_models_dir: str):
    """
    Scans the output directories specified in the configuration file and checks if the layers are complete, i.e. if they've passed
    the first phase of postprocessing and have a campaignout.csv and corresponding error model json file each.
    If 'ignore_incomplete_layers' in the configuration file is set to False, it raises an error as soon as an incomplete layer is
    found. Otherwise, it ignores any incomplete layer.
    As a final check, each network output directory should contain a netcontent.yaml file.

    If a network directory is ok, the error models' frequencies are adjusted and the results are saved to the initial models directory.
    The campaignout files are used to build a dataframe with the output frequencies for each layer. This dataframe is returned,
    so that it may be used for the following steps.
    """
    # postprocessing parameters
    ignore_incomplete: bool = config_dict['ignore_incomplete_layers']
    base_path = os.path.realpath(config_dict['outputs_base_path'])
    nvdla_configuration = config_dict['nvdla_config_id']
    networks_and_layers: dict = config_dict['networks_and_layers']

    # outputs for this step
    layer_df_rows = []

    for network, layers in networks_and_layers.items():
        network_dir = os.path.join(base_path, network, nvdla_configuration)
        if not os.path.isdir(network_dir):
            raise ValueError(f'Network output directory for {network} and config {nvdla_configuration} does not exist.')

        # check netcontent.yaml
        netcontent_path = os.path.join(network_dir, 'netcontent.yaml')
        if not os.path.exists(netcontent_path):
            raise ValueError(f'netcontent.yaml for network {network} does not exist.')
        
        # check individual layers
        for layer_name in layers:
            layer_path = os.path.join(network_dir, layer_name)
            if not os.path.isdir(layer_path):
                raise ValueError(f'Layer {layer_name} directory of network {network} does not exist.')
            
            # check campaignout.csv
            campaignout_path = os.path.join(layer_path, 'campaignout.csv')
            if not os.path.exists(campaignout_path):
                warning_msg = f'Layer {layer_name} of network {network} is missing its campaignout.csv file'
                if ignore_incomplete:
                    print(f'WARNING: {warning_msg} and will be skipped.')
                    continue
                else:
                    raise ValueError(warning_msg)
            
            # check error model
            classes_dir_path = os.path.join(layer_path, 'classes', 'classes')
            if not os.path.isdir(classes_dir_path):
                warning_msg = f'Layer {layer_name} of network {network} is missing the classes output directory'
                if ignore_incomplete:
                    print(f'WARNING: {warning_msg} and will be skipped.')
                    continue
                else:
                    raise ValueError(warning_msg)
                
            json_files = list(filter(
                lambda x: x.endswith('.json'),
                os.listdir(classes_dir_path)
            ))
            if not json_files or len(json_files) > 1:
                warning_msg = f'Layer {layer_name} of network {network} is missing the error model file or it has multiple json files'
                if ignore_incomplete:
                    print(f'WARNING: {warning_msg} and will be skipped.')
                    continue
                else:
                    raise ValueError(warning_msg)
            error_model_filepath = os.path.join(classes_dir_path, json_files[0])

            # get layer output frequencies from campaignout
            new_layer_df_row = utils.get_campaignout_last_row(campaignout_path, network, layer_name)
            layer_df_rows.append(new_layer_df_row)

            # load the error model and update its frequencies
            error_model = utils.replace_error_model_frequencies(error_model_filepath, new_layer_df_row)
            # save adjusted error model to directory
            error_model_name = f'{network}_{layer_name}.json'
            with open(os.path.join(initial_models_dir, error_model_name), 'w') as f:
                json.dump(error_model, f)

    # build complete layer dataframe
    layer_df = pd.DataFrame(layer_df_rows, index=['Layer'])
    return layer_df


if __name__ == '__main__':
    config_dict = load_config_dict()
    
    final_output_dir = os.path.realpath(config_dict['final_output_dir'])
    os.makedirs(final_output_dir, exist_ok=True)

    initial_models_dir = os.path.join(final_output_dir, 'initial_error_models')
    os.makedirs(initial_models_dir, exist_ok=True) # error models subdir

    output_filepath = os.path.join(final_output_dir, config_dict['step1_output_filename'])
    if not output_filepath.endswith('.xlsx'):
        output_filepath = output_filepath + '.xlsx'

    # build frequency dataframe
    layer_freq_df = check_layers_and_adjust_models(config_dict, initial_models_dir)

    # get layer hyperparameters from networks
    networks_and_layers: dict = config_dict['networks_and_layers']
    input_sizes: list = config_dict['network_input_sizes']
    layer_hyper_df = utils.build_hyperparameters_dataframe(networks_and_layers, input_sizes)

    # combine the two dataframes
    complete_df = pd.concat([layer_hyper_df, layer_freq_df], axis=1)

    # save dataframe
    print(complete_df)