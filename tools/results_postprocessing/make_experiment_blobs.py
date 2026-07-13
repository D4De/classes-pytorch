"""For each network, gathers experiment results in a single csv file."""
import os
import yaml
import pandas as pd

from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--output_dir', help='Path to the directory where to save the resulting blobs.')
    parser.add_argument('--classes_base_dir', help='Path to the experiments directory.')
    parser.add_argument('--hypers_base_dir', help='Path to the directory containing the network hyperparameters csv files.')
    parser.add_argument('--networks', nargs='+', help='Names of the networks.')
    parser.add_argument('--exp_dirs', nargs='+', help='Names of the experiment directories reachable from classes_base_dir (same order as networks).')
    parser.add_argument('--configs', nargs='+', help='Ids of the configurations. Should match the directory names in the experiments directories.')
    parser.add_argument('--spatial_classes', nargs='+', help='Names (Pascal) of the spatial classes to read in the applev files.')
    return parser.parse_args()


def snakecase_to_pascalcase(snake: str):
    """Utility class function to convert spatial class names from snake to pascal."""
    tokens = snake.split('_')
    return ''.join(token.capitalize() for token in tokens)


def extract_config_params(config: str):
    pieces = config.split('_')
    C, K = pieces[1].split('x')
    bitwidth = pieces[-1][3:]
    return int(C), int(K), int(bitwidth)


def find_default_file(candidate_filenames: list[str], beginning_token: str):
    possible_files = list(filter(lambda x: x.startswith(beginning_token), candidate_filenames))
    if len(possible_files) != 1:
        raise FileNotFoundError(f'Found either 0 or more than 1 possible "{beginning_token}" files in base directory. Provide a specific path.')

    return possible_files[0]


def main():
    args = parse_arguments()
    output_dir = args.output_dir
    classes_base_dir = args.classes_base_dir
    hypers_base_dir = args.hypers_base_dir
    networks = args.networks
    exp_dirs = args.exp_dirs
    configs = args.configs
    spatial_classes = args.spatial_classes

    if len(networks) != len(exp_dirs):
        raise ValueError('Lengths of networks and exp_dirs lists should be equal.')

    for network, exp_dir in zip(networks, exp_dirs):
        print(network)
        network_rows = []

        # load hyperparameter csv
        hypers_path = os.path.join(hypers_base_dir, f'{network}_hyper.csv')
        hypers_df = pd.read_csv(hypers_path, index_col=0)

        network_dir = os.path.join(classes_base_dir, exp_dir)

        for config in configs:
            C, K, bitwidth = extract_config_params(config)
            config_dir = os.path.join(network_dir, config)

            applev_filename = find_default_file(os.listdir(config_dir), 'applev')
            applev_path = os.path.join(config_dir, applev_filename)

            # load applev
            with open(applev_path) as f:
                applev_dict = yaml.load(f, yaml.SafeLoader)

            for i, (layer_name, layer_dict) in enumerate(applev_dict.items()):
                row_dict = {
                    'layer': layer_name,
                    'position': i,
                    'config': config,
                    'atomic-c': C,
                    'atomic-k': K,
                    'bitwidth': bitwidth,
                    'C': hypers_df.loc[layer_name, 'C'].item(),
                    'K': hypers_df.loc[layer_name, 'K'].item(),
                    'W': hypers_df.loc[layer_name, 'W'].item(),
                    'R': hypers_df.loc[layer_name, 'R'].item(),
                    'padding': hypers_df.loc[layer_name, 'padding'].item(),
                }

                # filter out non-spatial-classes
                class_names = [key for key in layer_dict if not key.startswith('prob')]
                # convert to pascal
                class_names_pascal = [snakecase_to_pascalcase(name) for name in class_names]

                # for each spatial class, if an entry is in the layer, take the critical frequency and multiply
                for spatial_class in spatial_classes:
                    if spatial_class in class_names_pascal:
                        class_name_snakecase = class_names[class_names_pascal.index(spatial_class)]
                        crit_freq = float(layer_dict[class_name_snakecase]['sdc_critical'])
                        row_dict[spatial_class] = crit_freq
                
                network_rows.append(pd.Series(row_dict))
        
        network_df = pd.concat(network_rows, axis=1)
        output_path = os.path.join(output_dir, f'{network}_application.csv')
        network_df.T.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()