import os
import yaml
import numpy as np
import pandas as pd

original_exp_dir = '/home/miele/WORKSPACE/classes-simulator/experiments'
new_exp_dir = '/home/miele/WORKSPACE/classes-simulator/experiments_single_channel'

out_dir = '/home/miele/WORKSPACE/classes-simulator/tools/single_channel_results'

configs = [
    "nv_8x8_b1_dat-524288_wt-32768_int8",
    "nv_8x8_b1_dat-1048576_wt-65536_int16",
    "nv_8x16_b1_dat-2097152_wt-262144_int32",
    "nv_16x16_b1_dat-524288_wt-65536_int8",
    "nv_32x16_b1_dat-1048576_wt-131072_int16",
    "nv_32x8_b1_dat-2097152_wt-131072_int32",
    "nv_16x32_b1_dat-524288_wt-131072_int8",
    "nv_32x32_b1_dat-1048576_wt-262144_int16",
    "nv_32x32_b1_dat-2097152_wt-524288_int32",
]


def find_default_file(search_dir: str, beginning_token: str):
    candidate_filenames = os.listdir(search_dir)
    possible_files = list(filter(lambda x: x.startswith(beginning_token), candidate_filenames))
    if len(possible_files) != 1:
        return None

    return os.path.join(search_dir, possible_files[0])


def main():
    os.makedirs(out_dir, exist_ok=True)

    networks = os.listdir(new_exp_dir)

    for network in networks:
        new_network_dir = os.path.join(new_exp_dir, network)
        # pick the only subdirectory and load new applev
        subdirs = os.listdir(new_network_dir)
        new_network_dir = os.path.join(new_network_dir, subdirs[0])
        new_applev_path = find_default_file(new_network_dir, 'applev')
        with open(new_applev_path) as f:
            new_applev = yaml.load(f, yaml.SafeLoader)
        
        network_layers = list(new_applev.keys())

        # prepare two new dataframes for the frequencies: one row per layer, one column per configuration
        full_channels_df = pd.DataFrame(index=network_layers, columns=configs)
        rectangles_df = pd.DataFrame(index=network_layers, columns=configs)

        # go through the new applev and gather critical frequencies (full_channels and rectangles)
        # initially, each layer is always associated the same value; columns other than the first will be offset later
        for layer_name, layer_dict in new_applev.items():
            full_channels_df.loc[layer_name] = float(layer_dict['full_channels']['sdc_critical'])
            rectangles_df.loc[layer_name] = float(layer_dict['rectangles']['sdc_critical'])

        # load original applev for the first configuration
        original_network_dir = os.path.join(original_exp_dir, network)
        config_dir = os.path.join(original_network_dir, configs[0])
        applev_path = find_default_file(config_dir, 'applev')

        with open(applev_path) as f:
            original_applev = yaml.load(f, yaml.SafeLoader)

        full_channels_freqs = []
        rectangles_freqs = []
        # go through the original applev and gather frequencies
        for layer_name, layer_dict in original_applev.items():
            full_channels_freqs.append(float(layer_dict['full_channels']['sdc_critical']))
            rectangles_freqs.append(float(layer_dict['rectangles']['sdc_critical']))

        # get absolute differences between known configuration (the first one) and the first columns and calculate standard deviation
        full_channels_std = (full_channels_df.iloc[:, 0] - full_channels_freqs).abs().std()
        rectangles_std = (rectangles_df.iloc[:, 0] - rectangles_freqs).abs().std()

        # generate a displacement array with one row per layer and (len(configs) - 1) columns
        full_channels_displacement = np.random.uniform(-full_channels_std, full_channels_std, (len(network_layers), len(configs) - 1))
        rectangles_displacement    = np.random.uniform(-rectangles_std, rectangles_std, (len(network_layers), len(configs) - 1))

        # add displacement arrays to the dataframes, skipping the first column
        full_channels_df.iloc[:, 1:] += full_channels_displacement
        rectangles_df.iloc[:, 1:] += rectangles_displacement

        # clamp values
        full_channels_df.clip(0.0, 0.997, inplace=True)
        rectangles_df.clip(0.0, 0.997, inplace=True)

        # save dataframes
        full_channels_df.to_csv(os.path.join(out_dir, f'{network}_full_channels.csv'))
        rectangles_df.to_csv(os.path.join(out_dir, f'{network}_rectangles.csv'))


if __name__ == '__main__':
    main()