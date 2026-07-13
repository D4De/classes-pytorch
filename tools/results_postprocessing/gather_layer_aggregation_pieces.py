import os
import yaml

try:
    # Per PyYAML docs: use the C-accelerated variants when available.
    from yaml import CSafeLoader as YLoader
except ImportError:
    from yaml import SafeLoader as YLoader

import pandas as pd

import scripts.postprocess.postproc_utils as utils

benchmarks_source_dir = '/home/miele/WORKSPACE/tcad2025/outdir'
results_path = '/home/miele/WORKSPACE/tcad2025/results-storage/network_reports/results.csv'
output_path = '/home/miele/WORKSPACE/tcad2025/results-storage/network_reports_aggregation_pieces.csv'

allowed_configs = [
    'nv_8x8_b1_dat-524288_wt-32768_int8',
    'nv_8x8_b1_dat-1048576_wt-65536_int16',
    'nv_32x8_b1_dat-2097152_wt-131072_int32',
    'nv_32x32_b1_dat-1048576_wt-262144_int16',
    'nv_16x16_b1_dat-524288_wt-65536_int8',
    'nv_16x32_b1_dat-524288_wt-131072_int8',
    'nv_32x16_b1_dat-1048576_wt-131072_int16',
    'nv_8x16_b1_dat-2097152_wt-262144_int32',
    'nv_32x32_b1_dat-2097152_wt-524288_int32',
]

def main():
    results_df = pd.read_csv(results_path, index_col=0)

    # filter configurations
    results_df = results_df[results_df['config'].isin(allowed_configs)]

    output_rows = []

    for layer_id, layer_row in results_df.groupby(['benchmark', 'layer', 'config']):
        benchmark, layer, config = layer_id

        output_layer_row = {
            'benchmark': benchmark,
            'layer': layer,
            'config': config,
        }

        # get campaignout
        campaignout_path = os.path.join(benchmarks_source_dir, 'benchmarks', benchmark, config, layer, 'campaignout.csv')
        if not os.path.exists(campaignout_path):
            print(f'WARNING: campaignout.csv missing for {benchmark} {config} {layer}')
            continue
            
        campaignout_df = pd.read_csv(campaignout_path, index_col=0)
        campaignout_df = campaignout_df.drop(['top'])

        # load other logs
        layerlog_path = os.path.join(benchmarks_source_dir, 'benchmarks', benchmark, config, f'{layer}.yaml')
        if not os.path.exists(layerlog_path):
            print(f'WARNING: layerlog missing for {benchmark} {config} {layer}')
            continue

        reglog_path = os.path.join(outdir, 'configs', config, f'{config}_reglog.yaml')
        if not os.path.exists(layerlog_path):
            print(f'WARNING: reglog missing for {benchmark} {config} {layer}')
            continue

        with open(layerlog_path) as f:
            layerlog = yaml.load(f, YLoader)
        with open(reglog_path) as f:
            reglog = yaml.load(f, YLoader)

        time_exposure = utils.get_timeexposure(layerlog)
        reg_exposure  = utils.get_regexposure(reglog)

        for unit, unit_row in campaignout_df.iterrows():
            output_layer_row[f'Silent-{unit}'] = unit_row['Silent']  # unit observability
            output_layer_row[f'area-{unit}'] = reg_exposure[unit]  # unit area factor
            output_layer_row[f'time-{unit}'] = time_exposure[unit] # unit time factor
        
        # add extra parameters
        output_layer_row['C'] = layer_row['C'].item()
        output_layer_row['K'] = layer_row['K'].item()
        output_layer_row['atomic-c'] = layer_row['atomic-c'].item()
        output_layer_row['atomic-k'] = layer_row['atomic-k'].item()
        output_layer_row['c_over_atomicc'] = layer_row['c_over_atomicc'].item()
        output_layer_row['k_over_atomick'] = layer_row['k_over_atomick'].item()
        output_layer_row['bitwidth'] = layer_row['bitwidth'].item()
        output_layer_row['fetch_over_compute'] = layer_row['fetch_over_compute'].item()
        output_layer_row['Silent'] = layer_row['Silent'].item()

        output_rows.append(output_layer_row)

    # build dataframe and save
    final_df = pd.DataFrame(output_rows)
    final_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()