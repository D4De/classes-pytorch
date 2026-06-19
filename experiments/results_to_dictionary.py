"""
Packages the results of an app-level error simulation in a dictionary saved to a yaml file, computing the total
probability of failure (and single probabilities of spatial class failures). This dictionary is known as an "applev" file.
Works on a single network experiment at a time.
"""

import os
import yaml

from argparse import ArgumentParser

from error_models.injection_campaign_postprocessing.postprocessing_utils import snakecase_spatial_classes

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('experiment_report_path', help='Filepath to the YAML experiment report produced by the simulation.')
    parser.add_argument('hw_config_id', help='Name of the hardware configuration from which the error models were obtained.')
    parser.add_argument('output_dir', help='Directory to save the resulting dictionary to.')
    return vars(parser.parse_args())


def sum_number_tuples(t1: tuple, t2: tuple):
    assert len(t1) == len(t2), f'The provided tuples have different lengths: {len(t1)} and {len(t2)}'
    return tuple(e1+e2 for e1,e2 in zip(t1,t2))


def pack_report_dictionary(report_dict: dict):
    layers_dict = {}

    for layer_name, layer_data in report_dict['Error simulation data']['Per-layer statistics'].items():
        layers_dict[layer_name] = {}
        spatial_classes_dict = {} # maps spatial classes to tuples of shape (num_masked, sdc_safe, sdc_critical)
        total_masked = total_sdc_safe = total_sdc_critical = 0

        # layer_data is another dictionary whose keys are error numbers; we are only interested in the values
        for error_data in layer_data.values():
            masked        = error_data['Masked']
            sdc_safe      = error_data['SDC safe']
            sdc_critical  = error_data['SDC critical']
            spatial_class = error_data['Spatial class']

            total_masked       += masked
            total_sdc_safe     += sdc_safe
            total_sdc_critical += sdc_critical

            if spatial_class in spatial_classes_dict:
                new_total = sum_number_tuples(spatial_classes_dict[spatial_class], (masked, sdc_safe, sdc_critical))
                spatial_classes_dict[spatial_class] = new_total
            else:
                spatial_classes_dict[spatial_class] = (masked, sdc_safe, sdc_critical)
        
        # compute result frequencies for each spatial class and save in dictionary
        for spatial_class, totals in spatial_classes_dict.items():
            final_total       = sum(totals)
            masked_freq       = float(totals[0] / final_total)
            sdc_safe_freq     = float(totals[1] / final_total)
            sdc_critical_freq = float(totals[2] / final_total)

            layers_dict[layer_name][spatial_class] = {
                'masked'       : masked_freq,
                'sdc_safe'     : sdc_safe_freq,
                'sdc_critical' : sdc_critical_freq,
            }
        
        # add entries with probability 0 for spatial classes that were not found in the report
        for spatial_class in snakecase_spatial_classes:
            if spatial_class not in layers_dict[layer_name]:
                layers_dict[layer_name][spatial_class] = {
                    'masked'       : 0.0,
                    'sdc_safe'     : 0.0,
                    'sdc_critical' : 0.0,
                }

        # add absolute frequencies to layer
        layer_total = total_masked + total_sdc_safe + total_sdc_critical
        layers_dict[layer_name]['prob_masked']       = float(total_masked/layer_total)
        layers_dict[layer_name]['prob_sdc_safe']     = float(total_sdc_safe/layer_total)
        layers_dict[layer_name]['prob_sdc_critical'] = float(total_sdc_critical/layer_total)
    
    return layers_dict


if __name__ == '__main__':
    args = parse_arguments()
    report_path = os.path.realpath(args['experiment_report_path'])
    if not os.path.exists(report_path):
        raise ValueError(f'Experiment report at {report_path} does not exist.')

    hw_config_name = args['hw_config_id']
    output_dir = os.path.realpath(args['output_dir'])

    # open report
    with open(report_path) as f:
        report_dict = yaml.load(f, yaml.SafeLoader)
    
    # prepare output
    network_name = report_dict['Model name']
    dataset_name = report_dict['Dataset data']['Dataset name']
    num_inputs   = report_dict['Experiment hyperparameters']['Batch size']
    num_errors   = report_dict['Experiment hyperparameters']['Injected errors per module']

    # Final filename will be
    # applev_<network>_<dataset>_<config>_<num_input>in_<num_error>err.yaml
    output_filename = f'applev_{network_name}_{dataset_name}_{hw_config_name}_{num_inputs}in_{num_errors}err.yaml'
    output_filepath = os.path.join(output_dir, output_filename)

    # loop through layer results
    layers_dict = pack_report_dictionary(report_dict)

    # save final dictionary
    with open(output_filepath, 'w') as f:
        yaml.dump(layers_dict, f, sort_keys=False)