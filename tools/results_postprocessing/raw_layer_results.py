"""
Given the results.csv data blob and the individual campaignout.csv files for each layer, gathers the relevant layer parameters
and frequencies into two csv files.
The parameters are a subset of those that characterize a layer within the context of a configuration, such as numbers of channels,
size of the input and of the kernel, AtomicC, AtomicK and so on.
The first file contains results for entire layers, that is frequencies that allow studying both their observability
(fraction of injections that resulted in an SDC) and susceptibility (weighted sum of the frequencies over the hardware units).
The second file acts as an observability "breakdown" and lists the raw frequencies for each hardware unit before rescaling.
"""
import os
import re
import yaml
import torch
import pandas as pd

from math import ceil

results_path = '/home/miele/WORKSPACE/results-storage/network_reports/results.csv'
benchmarks_path = '/home/miele/WORKSPACE/tcad2025/outdir/benchmarks'    # results of fault injection
classes_exp_path = '/home/miele/WORKSPACE/results-storage/error_simulation/experiments'
classes_extra_results_path = '/home/miele/WORKSPACE/results-storage/error_simulation/experiments_single_channel/single_channel_results' # single channel experiments
networks = [
    'alexnet_cifar10',
    'deeplabv3_oxfordpet',
    'mobilenetv2_gtsrb',
    'res50_cifar10',
    'res9_cifar10',
    'yolov11_coco',
]
network_hypers_path = '/home/miele/WORKSPACE/results-storage/network_hypers'

out_layer_path = '/home/miele/WORKSPACE/results-storage/network_reports/raw_layer_results.csv'
out_units_path = '/home/miele/WORKSPACE/results-storage/network_reports/raw_unit_results.csv'

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

def extract_config_parameters():
    cs = []
    ks = []
    bitwidths = []

    for config in configs:
        pieces = config.split('_')
        CxK = pieces[1]
        C, K = CxK.split('x')
        bitwidth = pieces[-1][3:]

        cs.append(int(C))
        ks.append(int(K))
        bitwidths.append(int(bitwidth))
    
    return cs, ks, bitwidths

cs, ks, bitwidths = extract_config_parameters()
units_to_skip = ['top', 'top.conv.dbuf'] # rows to avoid when scanning through campaignout files

def find_default_file(search_dir: str, beginning_token: str):
    candidate_filenames = os.listdir(search_dir)
    possible_files = list(filter(lambda x: x.startswith(beginning_token), candidate_filenames))
    if len(possible_files) != 1:
        raise FileNotFoundError(f'Found either 0 or more than 1 possible "{beginning_token}" files in {search_dir}. Provide a specific path.')

    return os.path.join(search_dir, possible_files[0])


def architectural_main():
    layer_df_rows = []
    units_df_rows = []

    reports_df = pd.read_csv(results_path, index_col=0)

    # columns that start with "class-" contain the final frequencies for the spatial class groups. Get their names
    spatial_group_names = list(filter(lambda x: x.startswith('class-'), reports_df.columns.tolist()))

    # filter configurations to avoid the extra ones
    reports_df = reports_df[reports_df['config'].isin(configs)]

    networks = reports_df['benchmark'].unique()

    for network in networks:
        network_rows = reports_df.query('benchmark == @network')
        layers = network_rows['layer'].unique()
        
        for config, C, K, bitwidth in zip(configs, cs, ks, bitwidths):
            config_rows = network_rows.query('config == @config')

            for layer in layers:
                campaignout_path = os.path.join(benchmarks_path, network, config, layer, 'campaignout.csv')

                if not os.path.exists(campaignout_path):
                    print(f'No campaignout at {campaignout_path}. Skipping.')
                    continue

                campaignout = pd.read_csv(campaignout_path, index_col=0)

                # drop unwanted rows
                campaignout = campaignout.drop(index=units_to_skip)
                # take SDC column's mean -> layer SDC observability
                layer_observability = float(campaignout['Silent'].mean())

                # take other hyperparameters from results.csv
                layer_row = config_rows.query('layer == @layer')
                if layer_row.empty:
                    print(f'Layer {layer} has no data. Skipping.')
                    continue

                new_layer_row = pd.Series({
                    'network': network,
                    'layer': layer,
                    'config': config,
                    'AtomicC': C,
                    'AtomicK': K,
                    'bitwidth': bitwidth,
                    'padding': layer_row['padding-x'].item(),
                    'C': layer_row['C'].item(),
                    'K': layer_row['K'].item(),
                    'W': layer_row['W'].item(),
                    'R': layer_row['R'].item(),
                    'TileSize': layer_row['TileSize'].item(),
                    'c_over_atomicc': layer_row['c_over_atomicc'].item(),
                    'k_over_atomick': layer_row['k_over_atomick'].item(),
                    'NumTiles': layer_row['NumTiles'].item(),
                    'fetch_over_compute': layer_row['fetch_over_compute'].item(),
                    'observability': layer_observability,
                    'susceptibility': layer_row['Silent'].item()
                })

                # add class group frequencies to the row
                spatial_groups_col = layer_row[spatial_group_names].squeeze(axis=0)
                new_layer_row = pd.concat([new_layer_row, spatial_groups_col])
                layer_df_rows.append(new_layer_row)

                # prepare rows for the hardware units
                for unit_name, unit_row in campaignout.iterrows():
                    new_unit_row = pd.Series({
                        'network': network,
                        'layer': layer,
                        'config': config,
                        'unit': unit_name,
                        'AtomicC': C,
                        'AtomicK': K,
                        'bitwidth': bitwidth,
                        'padding': layer_row['padding-x'].item(),
                        'C': layer_row['C'].item(),
                        'K': layer_row['K'].item(),
                        'W': layer_row['W'].item(),
                        'R': layer_row['R'].item(),
                        'TileSize': layer_row['TileSize'].item(),
                        'c_over_atomicc': layer_row['c_over_atomicc'].item(),
                        'k_over_atomick': layer_row['k_over_atomick'].item(),
                        'NumTiles': layer_row['NumTiles'].item(),
                        'observability': unit_row['Silent'],
                    })

                    units_df_rows.append(new_unit_row)
    
    final_layer_df = pd.concat(layer_df_rows, axis=1).T
    final_layer_df.to_csv(out_layer_path, index=False)

    final_units_df = pd.concat(units_df_rows, axis=1).T
    final_units_df.to_csv(out_units_path, index=False)


def get_freq_from_applev(applev_dict, layer: str, spatial_class: str):
    return float(applev_dict[layer][spatial_class]['sdc_critical'])


def average_freqs_from_applev(applev_dict, layer: str, spatial_classes: list[str]):
    total = 0
    for sp_class in spatial_classes:
        total += get_freq_from_applev(applev_dict, layer, sp_class)
    return float(total / len(spatial_classes))

def camel_to_snake(s: str):
    s = re.sub(r'(?<!^)([A-Z])', r'_\1', s)
    return s.lower()

def pattern_translator(name: str):
    if name == 'Skip4':
        return 'skip_4'
    else:
        return camel_to_snake(name)


def applev_main():
    device = torch.accelerator.current_accelerator()

    for network in networks:
        print(f'Gathering results for {network}')

        output_path = os.path.join('outdir', f'raw_{network}_results.csv')
        new_rows = []

        # get hyperparameters
        print('Getting hyperparameters')
        hyper_df = pd.read_csv(os.path.join(network_hypers_path, f'{network}_hyper.csv'), index_col=0)

        # get FIT report
        print('Loading FIT report')
        with open(os.path.join('outdir', 'reports', f'{network}_final.yaml')) as f:
            fit_dict = yaml.load(f, yaml.SafeLoader)

        # get list of layers from report
        for config_dict in fit_dict.values():
            layers = list(config_dict['layers'].keys())
            break

        # load reports for extra single channel critical frequencies results
        single_full_channels_crit_df = pd.read_csv(os.path.join(classes_extra_results_path, f'exp_{network}_full_channels.csv'), index_col=0)
        single_rectangles_crit_df = pd.read_csv(os.path.join(classes_extra_results_path, f'exp_{network}_rectangles.csv'), index_col=0)

        for config, C, K, bitwidth in zip(configs, cs, ks, bitwidths):
            print(f'Starting configuration {config}')
            classes_exp_dir = os.path.join(classes_exp_path, f'exp_{network}', config)

            print('Loading SDC frequencies')
            # load SDC frequencies file
            sdc_freq_path = find_default_file(classes_exp_dir, 'SDC_')
            sdc_freqs = pd.read_excel(sdc_freq_path, index_col=0)

            print('Loading applev')
            # load applev
            applev_path = find_default_file(classes_exp_dir, 'applev_')
            with open(applev_path) as f:
                applev_dict = yaml.load(f, yaml.SafeLoader)

            # get FIT dictionary for current config
            current_fit_dict = fit_dict[config]['layers']

            for layer in layers:
                print(f'Starting layer {layer}')
                
                print('Getting layer FIT')
                layer_fit = float(current_fit_dict[layer]['FIT'])
        
                # get layer parameters
                layer_hypers = hyper_df.loc[layer].squeeze()

                new_layer_row = pd.Series({
                    'layer': layer,
                    'config': config,
                    'AtomicC': C,
                    'AtomicK': K,
                    'bitwidth': bitwidth,
                    'padding': layer_hypers['padding'].item(),
                    'C': layer_hypers['C'].item(),
                    'K': layer_hypers['K'].item(),
                    'W': layer_hypers['W'].item(),
                    'R': layer_hypers['R'].item(),
                    'c_over_atomicc': ceil(layer_hypers['C'].item() / C),
                    'k_over_atomick': ceil(layer_hypers['K'].item() / K),
                    'SDC': float(sdc_freqs.at[layer, 'SDC']),
                    'class-Single_crit': get_freq_from_applev(applev_dict, layer, 'single'),
                    'class-SingleChannelRandom_crit': average_freqs_from_applev(applev_dict, layer, ['skip_4', 'single_channel_alternated_blocks', 'single_channel_random']),
                    'class-SingleChannelBlock_crit': (get_freq_from_applev(applev_dict, layer, 'single_block') + get_freq_from_applev(applev_dict, layer, 'same_row') + float(single_rectangles_crit_df.at[layer, config])) / 3.0,
                    'class-SingleFullChannel_crit': float(single_full_channels_crit_df.at[layer, config]),
                    'class-MultiChannelRandom_crit': average_freqs_from_applev(applev_dict, layer, ['skip_4', 'shattered_channel']),
                    'class-MultiFullChannels_crit': get_freq_from_applev(applev_dict, layer, 'full_channels'),
                    'class-MultiChannelBlock_crit': average_freqs_from_applev(applev_dict, layer, ['same_row', 'multi_channel_block', 'rectangles']),
                    'class-BulletWake_crit': get_freq_from_applev(applev_dict, layer, 'bullet_wake'),
                    'FIT': layer_fit,
                })

                new_rows.append(new_layer_row)

        final_network_df = pd.concat(new_rows, axis=1).T
        final_network_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    architectural_main()
    applev_main()