import os
import yaml
import pandas as pd

from argparse import ArgumentParser

def load_config_file():
    parser = ArgumentParser()
    parser.add_argument('args_file_path', help='Path to the rescaling args YAML file.')
    args = parser.parse_args()

    args_file_path = os.path.realpath(args.args_file_path)
    if not os.path.exists(args_file_path):
        raise FileNotFoundError(f'Cannot find args file {args_file_path}.')
    
    with open(args_file_path) as f:
        args_dict: dict = yaml.load(f, yaml.SafeLoader)
    
    return args_dict


def find_default_file(candidate_filenames: list[str], beginning_token: str):
    possible_files = list(filter(lambda x: x.startswith(beginning_token), candidate_filenames))
    if len(possible_files) != 1:
        raise FileNotFoundError(f'Found either 0 or more than 1 possible "{beginning_token}" files in base directory. Provide a specific path.')

    return possible_files[0]


def snakecase_to_pascalcase(snake: str):
    tokens = snake.split('_')
    return ''.join(token.capitalize() for token in tokens)


if __name__ == '__main__':
    args_dict = load_config_file()

    base_path      = os.path.realpath(args_dict['experiment_base_path'])
    out_excel_path = os.path.realpath(args_dict['output_excel_path'])

    random_spatial_classes: bool = args_dict['random_spatial_classes']

    network_dataset_ids = args_dict['network_dataset_ids']
    configuration_ids   = args_dict['configuration_ids']    
    short_config_ids    = args_dict['short_configuration_ids']

    # The result of this script is an Excel file with 3 sheets: spatial classes vulnerability distribution, generale layer metrics and
    # vulnerability values. Each sheet is modeled with a MultiIndex DataFrame.
    
    # Spatial classes:
    #                 | Config1 | Config2 | ...
    # Class | Network |
    sp_class_row_index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['Class', 'Network'])
    sp_class_df = pd.DataFrame(index=sp_class_row_index, columns=short_config_ids)

    # Metrics:
    #                 | Config1    | Config2    | ...
    #                 | Metrics... | Metrics... | ...
    # Network | Layer |
    metrics_row_index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['Network', 'Layer'])
    metrics_col_index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['Config', 'Metrics'])
    metrics_df = pd.DataFrame(index=metrics_row_index, columns=metrics_col_index)

    # Vulnerability:
    #                 | Config1 | Config2 | ...
    # Network | Layer |
    vuln_row_index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['Network', 'Layer'])
    vuln_df = pd.DataFrame(index=vuln_row_index, columns=short_config_ids)

    # CHECK PATHS AND LOAD FILES

    if not os.path.isdir(base_path):
        raise ValueError(f'Base experiment directory at {base_path} does not exist.')

    if not out_excel_path.endswith('.xlsx'):
        out_excel_path = out_excel_path + '.xlsx'

    # for each spatial class, track the total critical error probability per network and per configuration
    sp_class_crit_probs: dict[str, dict[str, dict[str, float]]] = {} # class -> network -> config 

    for network_id in network_dataset_ids:
        for config_id, short_id in zip(configuration_ids, short_config_ids):
            exp_dir = os.path.join(base_path, f'exp_{network_id}', config_id)
            if not os.path.exists(exp_dir):
                raise ValueError(f'Network directory {exp_dir} does not exist.')
            
            # look for default files
            network_dir_files = list(filter(lambda x: not os.path.isdir(x), os.listdir(exp_dir)))

            applev_path     = os.path.join(exp_dir, find_default_file(network_dir_files, 'applev'))
            sdc_freq_path   = os.path.join(exp_dir, find_default_file(network_dir_files, 'SDC'))
            netcontent_path = os.path.join(exp_dir, find_default_file(network_dir_files, 'netcontent'))

            # existence checks
            if not os.path.exists(applev_path):
                raise FileNotFoundError(f'applev report file at {applev_path} does not exist.')
            if not os.path.exists(sdc_freq_path):
                raise FileNotFoundError(f'SDC frequencies Excel file at {sdc_freq_path} does not exist.')
            if not os.path.exists(netcontent_path):
                raise FileNotFoundError(f'Netcontent YAML file at {netcontent_path} does not exist.')

            # load files
            with open(applev_path) as f:
                applev_dict: dict = yaml.load(f, yaml.SafeLoader)
            with open(netcontent_path) as f:
                netcontent_dict = yaml.load(f, yaml.SafeLoader)
            netcontent_modules: dict = netcontent_dict['modules']
            
            sdc_freqs = pd.read_excel(sdc_freq_path, sheet_name=0, index_col=0, header=0)

            # COMPUTE METRICS

            total_network_vuln = 0.0
            num_layers = len(applev_dict.keys())

            # check that all 3 layer lists are the same and create the rows
            for layer_name, layer_dict in applev_dict.items():
                if layer_name not in netcontent_modules.keys():
                    raise ValueError(f'Layer {layer_name} is in the applev report, but not in the netcontent file.')
                if layer_name not in sdc_freqs.index:
                    raise ValueError(f'Layer {layer_name} is in the applev report, but not in the SDC frequencies file.')

                # get layer time exposure
                time_exposure = float(netcontent_modules[layer_name]['seu-prob'])
                # get layer SDC frequency
                layer_sdc_freq = float(sdc_freqs.at[layer_name, 'SDC'])

                # for each layer, track the total and scaled outcome probabilities
                total_app_masked_prob = total_app_safe_prob = total_app_critical_prob = 0.0

                for spatial_class_name, spatial_class_dict in layer_dict.items():
                    # skip non-spatial classes
                    if spatial_class_name.startswith('prob'):
                        continue

                    # determine scale factor for the class
                    if random_spatial_classes:
                        scale_factor = 1.0
                    else:
                        class_name_pascal = snakecase_to_pascalcase(spatial_class_name)
                        if class_name_pascal in sdc_freqs.columns:
                            scale_factor = float(sdc_freqs.at[layer_name, class_name_pascal])
                        else:
                            scale_factor = 0.0

                    # scale probabilities
                    app_masked_prob   = spatial_class_dict['masked']
                    app_safe_prob     = spatial_class_dict['sdc_safe']
                    app_critical_prob = spatial_class_dict['sdc_critical']

                    # add critical probability to overall dictionary, averaging it over all the layers
                    if spatial_class_name not in sp_class_crit_probs: # add spatial class dictionary if necessary
                        sp_class_crit_probs[spatial_class_name] = {}
                    if network_id not in sp_class_crit_probs[spatial_class_name]: # add network dictionary if necessary
                        sp_class_crit_probs[spatial_class_name][network_id] = {}
                    if short_id not in sp_class_crit_probs[spatial_class_name]: # initialize configuration count if necessary
                        sp_class_crit_probs[spatial_class_name][network_id][short_id] = 0.0

                    sp_class_crit_probs[spatial_class_name][network_id][short_id] += app_critical_prob / num_layers

                    # update total probabilities
                    total_app_masked_prob   += scale_factor * app_masked_prob
                    total_app_safe_prob     += scale_factor * app_safe_prob
                    total_app_critical_prob += scale_factor * app_critical_prob

                # rescale total layer frequencies by SDC frequency (probability for the SEU to affect the layer output)
                arch_masked_prob   = (1.0 - layer_sdc_freq) * total_app_masked_prob
                arch_safe_prob     = layer_sdc_freq * total_app_safe_prob
                arch_critical_prob = layer_sdc_freq * total_app_critical_prob

                # rescale frequencies by time exposure (probability for a SEU to strike the layer)
                time_masked_prob   = (1.0 - time_exposure) * arch_masked_prob 
                time_safe_prob     = time_exposure * arch_safe_prob 
                time_critical_prob = time_exposure * arch_critical_prob

                # build metrics row
                metrics_df.at[(network_id, layer_name), (short_id, 'Time exposure')]           = time_exposure
                metrics_df.at[(network_id, layer_name), (short_id, 'SDC freq')]                = layer_sdc_freq
                metrics_df.at[(network_id, layer_name), (short_id, 'Masked freq')]             = total_app_masked_prob
                metrics_df.at[(network_id, layer_name), (short_id, 'Safe freq')]               = total_app_safe_prob
                metrics_df.at[(network_id, layer_name), (short_id, 'Critical freq')]           = total_app_critical_prob
                metrics_df.at[(network_id, layer_name), (short_id, 'Vuln(crit * sdc)')]        = arch_critical_prob 
                metrics_df.at[(network_id, layer_name), (short_id, 'Vuln(crit * sdc * time)')] = time_critical_prob 

                # record vulnerability value for the layer
                vuln_df.at[(network_id, layer_name), short_id] = time_critical_prob
                total_network_vuln += time_critical_prob
            
            # record total network vulnerability
            vuln_df.at[(network_id, 'TOTAL'), short_id] = total_network_vuln

    # build spatial class df
    for class_name, class_dict in sp_class_crit_probs.items():
        for network_id, network_dict in class_dict.items():
            for short_id, crit_prob in network_dict.items():
                sp_class_df.at[(class_name, network_id), short_id] = crit_prob

    # SAVE TO FILE
    with pd.ExcelWriter(out_excel_path, mode='w', engine='openpyxl') as writer:
        sp_class_df.to_excel(writer, sheet_name='SpatialClasses', index_label=['Class', 'Network'])
        metrics_df.to_excel(writer, sheet_name='LayerMetrics', index_label=['Network', 'Layer'])
        vuln_df.to_excel(writer, sheet_name='Vulnerability', index_label=['Network', 'Layer'])
