import os
import yaml
import numpy as np
import pandas as pd

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('models_base_dir', help='Directory containing the models subdirectories. They should all start with models_')
    parser.add_argument('results_filepath', help='Path to results.csv')
    return parser.parse_args()


def split_network_and_layer(complete_name: str):
    pieces = complete_name.split(sep='_')
    network = '_'.join([pieces[0], pieces[1]])
    layer = pieces[2]
    return network, layer


def main():
    args = parse_arguments()
    models_base_dir = os.path.realpath(args.models_base_dir)
    results_filepath = os.path.realpath(args.results_filepath)

    if not os.path.isdir(models_base_dir):
        raise ValueError(f'Models directory at {models_base_dir} does not exist.')
    if not os.path.exists(results_filepath):
        raise FileNotFoundError(f'results.csv file at {results_filepath} does not exist.')

    results_df = pd.read_csv(results_filepath)

    models_dirs = [os.path.join(models_base_dir, name) for name in os.listdir(models_base_dir) if name.startswith('models_')]

    for models_dir in models_dirs:
        config_filepath = os.path.join(models_dir, 'postprocessing_config.yaml')
        step1_df_filepath = os.path.join(models_dir, 'step1_complete_df.xlsx')

        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f'Config file at {config_filepath} does not exist.')
        if not os.path.exists(step1_df_filepath):
            raise FileNotFoundError(f'Step 1 df file at {step1_df_filepath} does not exist.')

        with open(config_filepath) as f:
            config_dict = yaml.load(f, yaml.SafeLoader)
            configuration_id = config_dict['nvdla_config_id']
        
        step1_df = pd.read_excel(step1_df_filepath)
        # take only Layer and SDC columns
        step1_df = step1_df.loc[:, ['Layer', 'SDC']].set_index('Layer')

        for layer_name, series in step1_df.iterrows():
            network, layer = split_network_and_layer(layer_name)
            sdc = float(series['SDC'])

            other_sdc_row = results_df.loc[ (results_df['benchmark'] == network) & (results_df['layer'] == layer) & (results_df['config'] == configuration_id)]
            other_sdc = float(other_sdc_row.iloc[0]['Silent'].item())
            
            if not np.isclose(sdc, other_sdc):
                print(f'WARNING: SDC values in {models_dir} and results.csv differ for {network}, {layer}, {configuration_id}')


if __name__ == '__main__':
    main()