import os
import json
import shutil
import pandas as pd

import error_models.injection_campaign_postprocessing.postprocessing_utils as utils


def check_layers(outputs_base_path: str, nvdla_configuration: str, networks_and_layers: dict):
    """
    Scans the output directories specified in the configuration file and checks if the layers are complete, i.e. if they've passed
    the first phase of postprocessing and have a campaignout.csv and corresponding error model json file each.
    Each network output directory should also contain a netcontent.yaml file.
    
    An error is raised as soon as an incomplete layer is found.
    If all layers are complete, the function returns a dictionary with the following structure:\\
    network_name:\\
        - netcontent_path\\
        - layers:\\
            -- layer_name:\\
                --- campaignout_path\\
                --- error_model_path
    """
    final_paths = {}

    for network, layers in networks_and_layers.items():
        network_dir = os.path.join(outputs_base_path, network, nvdla_configuration)

        # check existence of network directory
        if not os.path.isdir(network_dir):
            raise ValueError(f'Network output directory for {network} and config {nvdla_configuration} does not exist.')

        # check existence of netcontent.yaml
        netcontent_path = os.path.join(network_dir, 'netcontent.yaml')
        if not os.path.exists(netcontent_path):
            raise ValueError(f'netcontent.yaml for network {network} does not exist.')

        # prepare network entry
        final_paths[network] = {
            'netcontent_path': netcontent_path,
            'layers': {},
        }

        # check individual layers
        for layer_name in layers:
            layer_dir = os.path.join(network_dir, layer_name)

            # check existence of layer directory
            if not os.path.isdir(layer_dir):
                raise ValueError(f'Layer {layer_name}\'s directory for network {network} does not exist.')

            # check existence of campaignout.csv
            campaignout_path = os.path.join(layer_dir, 'campaignout.csv')
            if not os.path.exists(campaignout_path):
                raise ValueError(f'Layer {layer_name} of network {network} is missing its campaignout.csv file')
            
            # check existence of error model directory
            classes_dir_path = os.path.join(layer_dir, 'classes', 'classes')
            if not os.path.isdir(classes_dir_path):
                raise ValueError(f'Layer {layer_name} of network {network} is missing the classes output directory.')
            
            # check existence of error model
            json_files = list(filter(
                lambda x: x.endswith('.json'),
                os.listdir(classes_dir_path)
            ))
            if not json_files or len(json_files) > 1:
                raise ValueError(f'Layer {layer_name} of network {network} is missing the error model file or it has multiple json files.')
            error_model_filepath = os.path.join(classes_dir_path, json_files[0])

            # add layer entry
            final_paths[network]['layers'][layer_name] = {
                'campaignout_path': campaignout_path,
                'error_model_path': error_model_filepath,
            }

    return final_paths


def build_layer_df_and_adjust_models(filepaths_dict: dict, initial_models_dir: str):
    """
    All campaignout files that were found are opened and processed to produce the first complete layer dataframe. At the same
    time, the frequency data in the campaignout files is used to adjust the frequencies within the error models.
    The new error models are saved in the provided directory.
    """
    layer_df_rows = []

    for network_name, network_dict in filepaths_dict.items():
        for layer_name, layer_paths in network_dict['layers'].items():
            layer_id = f'{network_name}_{layer_name}'

            # get layer output frequencies from campaignout
            new_layer_df_row = pd.read_csv(layer_paths['campaignout_path']).iloc[-1]
            # set layer id
            new_layer_df_row['Unit'] = layer_id
            layer_df_rows.append(new_layer_df_row)

            # load the error model and update its frequencies
            error_model = replace_error_model_frequencies(layer_paths['error_model_path'], new_layer_df_row)
            # save adjusted error model to directory
            error_model_name = layer_id + '.json'
            with open(os.path.join(initial_models_dir, error_model_name), 'w') as f:
                json.dump(error_model, f)

    # build complete layer dataframe
    layer_df = pd.DataFrame(layer_df_rows)

    # rename columns
    layer_df.rename(columns={'Unit': 'Layer', 'Silent': 'SDC'}, inplace=True)
    layer_df.set_index('Layer', inplace=True)

    # sum crash and hang
    layer_df['Crash+Hang'] = layer_df['SegFault'] + layer_df['Timeout']
    layer_df.drop(columns=['SegFault', 'Timeout'], inplace=True)

    # drop columns of extra spatial classes
    spatial_freqs = layer_df.drop(columns=['Masked', 'SDC', 'Crash+Hang'])
    for label in spatial_freqs:
        if label not in utils.spatial_classes:
            layer_df.drop(columns=[label], inplace=True)
    
    # add missing spatial classes
    for label in utils.spatial_classes:
        if label not in layer_df:
            layer_df[label] = 0.0

    # replace NaN with 0
    layer_df.fillna(0.0, inplace=True)

    # change column order and set index
    layer_df = layer_df.reindex(columns=['Masked', 'SDC', 'Crash+Hang'] + utils.spatial_classes)

    return layer_df


def replace_error_model_frequencies(error_model_filepath: str, new_frequencies: pd.Series) -> dict:
    with open(error_model_filepath) as f:
        error_model = json.load(f)
    
    # discard non-frequency items
    new_frequencies_filtered = new_frequencies.drop(['Unit', 'Silent', 'SegFault', 'Timeout'])

    for spatial_class, frequency in new_frequencies_filtered.items():
        spatial_class_snake = utils.camelcase_to_snakecase(spatial_class)
        # it's possible that a spatial class is not in the error model because it was excluded
        if spatial_class_snake in error_model:
            error_model[spatial_class_snake]['frequency'] = frequency
    
    return error_model


def copy_netcontent_files(filepaths_dict: dict, netcontent_dir: str):
    for network_name, network_dict in filepaths_dict.items():
        # make network directory
        network_dir = os.path.join(netcontent_dir, network_name)
        os.makedirs(network_dir, exist_ok=True)

        src_path = network_dict['netcontent_path']
        dst_path = os.path.join(network_dir, 'netcontent.yaml')
        shutil.copyfile(src_path, dst_path)


def compute_zscores(complete_df: pd.DataFrame):
    """Starting from the complete layer dataframe, builds a new one with the Z-score computed for each frequency column.
    Also groups layer rows with the same hyperparameters and computes the Z-scores within each group. Returns both dataframes."""

    def _compute_row_group_zscores(group_df: pd.DataFrame):
        """
        Given a subset of a dataframe's rows, splits frequency columns from non-frequency ones. Then, for each frequency
        column, the corresponding z-score column is computed. The two types of columns are interleaved and pre-concatenated
        with the non-frequency columns. The resulting dataframe is returned.
        """
        non_freq_df: pd.DataFrame = group_df[group_df.columns.difference(utils.macroscopic_results + utils.spatial_classes)]
        freq_df: pd.DataFrame = group_df[utils.macroscopic_results + utils.spatial_classes]

        # interleave frequency columns and z-score columns
        new_cols = []

        for spatial_class_name in freq_df:
            freq_col = freq_df[spatial_class_name]
            # compute z-scores for the column
            zscore_col = (freq_col - freq_col.mean()) / freq_col.std()
            zscore_col.name = 'Z-' + zscore_col.name

            new_cols.append(freq_col)
            new_cols.append(zscore_col)

        zscore_df = pd.concat([
            non_freq_df,
            pd.DataFrame(new_cols).T,
        ], axis=1)

        return zscore_df

    # use the entire dataframe as a group to compute the first set of columns
    complete_zscored_df = _compute_row_group_zscores(complete_df)

    # group the rows which share the same hyperparameters and compute zscores for each group
    group_dfs = []
    for group_layers, group in complete_df.groupby(by=utils.hyperparameters):
        group_df = _compute_row_group_zscores(group)
        group_dfs.append(group_df)
    grouped_zscored_df = pd.concat(group_dfs)

    return complete_zscored_df, grouped_zscored_df


if __name__ == '__main__':
    config_dict = utils.load_config_dict()
    
    outputs_base_path = os.path.realpath(config_dict['outputs_base_path'])

    final_output_dir = os.path.realpath(config_dict['final_output_dir'])
    os.makedirs(final_output_dir, exist_ok=True)

    initial_models_dir = os.path.join(final_output_dir, utils.initial_models_dirname)
    os.makedirs(initial_models_dir, exist_ok=True)

    netcontent_dir = os.path.join(final_output_dir, utils.netcontent_dirname)
    os.makedirs(netcontent_dir, exist_ok=True)

    output_filepath = os.path.join(final_output_dir, utils.step1_output_filename)

    # check layers and get filepaths dictionary
    filepaths_dict = check_layers(outputs_base_path, config_dict['nvdla_config_id'], config_dict['networks_and_layers'])

    # build frequency dataframe
    layer_freq_df = build_layer_df_and_adjust_models(filepaths_dict, initial_models_dir)

    # copy netcontent files
    copy_netcontent_files(filepaths_dict, netcontent_dir)

    # get layer hyperparameters from networks
    layer_hyper_df = utils.build_hyperparameters_dataframe(config_dict['networks_and_layers'], config_dict['network_input_sizes'])

    # combine the two dataframes
    complete_df = pd.concat([layer_hyper_df, layer_freq_df], axis=1)

    # compute z-scores
    complete_zscored_df, grouped_zscored_df = compute_zscores(complete_df)

    # save dataframes
    with pd.ExcelWriter(output_filepath, mode='w') as writer:
        complete_df.to_excel(writer, sheet_name='complete_layers')
        complete_zscored_df.to_excel(writer, sheet_name='zscored_layers')
        grouped_zscored_df.to_excel(writer, sheet_name='grouped_zscored_layers')

    print('STEP 1 DONE')
    print(f'The complete layer dataframe has been saved as an Excel file to {output_filepath}.')
    print('Sheets beyond the first one contain already-calculated Z-scores to help identify outliers.')
    print('Examine the file and remove outlier rows from the "complete_layers" sheet. When done, save the file and proceed with step 2.')