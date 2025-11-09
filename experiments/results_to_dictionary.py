import os
import yaml

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('experiment_report_path', help='Filepath to the YAML experiment report produced by the simulation.')
    parser.add_argument('hw_config_id', help='Name of the hardware configuration from which the error models were obtained.')
    parser.add_argument('output_path', help='Filepath to save the resulting dictionary to.')
    return vars(parser.parse_args())

def sum_number_tuples(t1: tuple, t2: tuple):
    assert len(t1) == len(t2), f'The provided tuples have different lengths: {len(t1)} and {len(t2)}'
    return tuple(e1+e2 for e1,e2 in zip(t1,t2))

if __name__ == '__main__':
    args = parse_arguments()
    report_path = os.path.realpath(args['experiment_report_path'])
    if not os.path.exists(report_path):
        raise ValueError(f'Experiment report at {report_path} does not exist.')

    hw_config_name = args['hw_config_id']
    output_path = os.path.realpath(args['output_path'])

    # open report
    with open(report_path) as f:
        report_dict = yaml.load(f, yaml.SafeLoader)
    
    # prepare output
    network_id = f'{report_dict['Model name']}_{report_dict['Dataset data']['Dataset name']}'
    batch_size = int(report_dict['Experiment hyperparameters']['Batch size'])
    num_errors = int(report_dict['Experiment hyperparameters']['Injected errors per module'])

    output_dict = {
        hw_config_name: {
            network_id: {
                'input_samples': batch_size,
                'errors_per_layer': num_errors,
            }
        }
    }

    # loop through layer results
    layers_dict = {}
    for layer_name, layer_data in report_dict['Error simulation data']['Per-layer statistics'].items():
        layers_dict[layer_name] = {}
        spatial_classes_dict = {} # maps spatial classes to tuples of shape (num_masked, sdc_safe, sdc_critical)

        # layer_data is another dictionary whose keys are error numbers; we are only interested in the values
        for error_data in layer_data.values():
            masked = error_data['Masked']
            sdc_safe = error_data['SDC safe']
            sdc_critical = error_data['SDC critical']
            spatial_class = error_data['Spatial class']

            if spatial_class in spatial_classes_dict:
                new_total = sum_number_tuples(spatial_classes_dict[spatial_class], (masked, sdc_safe, sdc_critical))
                spatial_classes_dict[spatial_class] = new_total
            else:
                spatial_classes_dict[spatial_class] = (masked, sdc_safe, sdc_critical)
        
        # compute result frequencies for each spatial class and save in dictionary
        for spatial_class, totals in spatial_classes_dict.items():
            final_total = sum(totals)
            masked_freq = float(totals[0] / final_total)
            sdc_safe_freq = float(totals[1] / final_total)
            sdc_critical_freq = float(totals[2] / final_total)

            layers_dict[layer_name][spatial_class] = {
                'masked': masked_freq,
                'sdc_safe': sdc_safe_freq,
                'sdc_critical': sdc_critical_freq,
            }

    output_dict[hw_config_name][network_id]['layer_results'] = layers_dict

    # save final dictionary
    with open(output_path, 'w') as f:
        yaml.dump(output_dict, f)