"""
Takes the output report of an experiment and the 'netcontent.yaml' file of the corresponding network (obtained via network profilation
prior to architectural-level injection), extracts the time exposures from the latter and computes the final masked and SDC frequencies
for each layer in the simulation.
"""
import os
import yaml
import pandas as pd

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('report_path', help='Filepath to the experiment output report.')
    parser.add_argument('netcontent_path', help='File path to the netcontent.yaml file for the network used in the experiment.')
    parser.add_argument('output_path', help='File path to save the resulting file to.')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_arguments()
    report_path = args['report_path']
    netcontent_path = args['netcontent_path']
    output_path = args['output_path']

    if not os.path.exists(report_path):
        raise ValueError(f'Report at {report_path} does not exist.')
    if not os.path.exists(netcontent_path):
        raise ValueError(f'Netcontent file at {netcontent_path} does not exist.')
    
    with open(report_path) as f:
        report_dict = yaml.load(f, yaml.SafeLoader)
    with open(netcontent_path) as f:
        netcontent_dict = yaml.load(f, yaml.SafeLoader)

    # get time exposures from netcontent
    layer_time_exposures: dict[str, float] = {}
    for layer_name, info in netcontent_dict['modules'].items():
        if info['optype'] == 'conv':
            layer_time_exposures[layer_name] = info['seu-prob']

    # compute result frequencies for all layers
    layer_result_frequencies = []
    for layer_name, results in report_dict['Error simulation data']['Per-layer statistics'].items():
        time_exposure = layer_time_exposures.get(layer_name, 1.0)
        masked = sdc_safe = sdc_critical = 0

        for single_error_results in results.values():
            masked += single_error_results['Masked']
            sdc_safe += single_error_results['SDC safe']
            sdc_critical += single_error_results['SDC critical']
        
        total = masked + sdc_safe + sdc_critical

        masked_freq = float(masked / total)
        sdc_safe_freq = float(sdc_safe / total)
        sdc_critical_freq = float(sdc_critical / total)

        masked_adj = masked_freq * time_exposure
        sdc_safe_adj = sdc_safe_freq * time_exposure
        sdc_critical_adj = sdc_critical_freq * time_exposure
        total_sdc_freq = sdc_safe_adj + sdc_critical_adj

        layer_result_frequencies.append({
            'Layer': layer_name,
            'Time exposure': time_exposure,
            'Masked': masked_freq,
            'SDC safe': sdc_safe_freq,
            'SDC critical': sdc_critical_freq,
            'Masked Adj': masked_adj,
            'SDC safe Adj': sdc_safe_adj,
            'SDC critical Adj': sdc_critical_adj,
            'Total SDC Adj': total_sdc_freq,
        })
    
    # build dataframe of results
    df = pd.DataFrame(layer_result_frequencies,
                      columns=['Layer', 'Time exposure', 'Masked', 'SDC safe', 'SDC critical', 'Masked Adj', 'SDC safe Adj', 'SDC critical Adj', 'Total SDC Adj'])
    
    if not output_path.endswith('.csv'):
        output_path = output_path + '.csv'
    df.to_csv(output_path, index=False)