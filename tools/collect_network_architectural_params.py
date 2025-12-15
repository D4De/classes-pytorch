import os
import yaml
import pandas as pd

from argparse import ArgumentParser

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('network_base_path', help='Path to the output directory for the network, containing netcontent.yaml and all the' \
        'layerlog files for the network layers.')
    parser.add_argument('network_id', help='Example: resnet50_cifar10. Used for labeling.')
    parser.add_argument('output_path', help='Filepath to the output Excel file that will be produced.')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_arguments()
    base_path = os.path.realpath(args['network_base_path'])
    if not os.path.isdir(base_path):
        raise ValueError(f'Network base path at {base_path} is not a directory.')
    network_id = args['network_id']
    output_path = os.path.realpath(args['output_path'])

    dir_filenames: list = os.listdir(base_path)
    dir_filenames.remove('netcontent.yaml') # ignore netcontent if present

    df_rows = []
    for filename in dir_filenames:
        filepath = os.path.join(base_path, filename)

        # skip directories
        if os.path.isdir(filepath):
            continue

        # load layerlog
        with open(filepath) as f:
            layerlog = yaml.load(f, yaml.SafeLoader)

        layer_name = filename[:-5] # strip yaml extension

        # numbers of tiles
        num_macro_tiles = len(layerlog['tiles'].keys())
        first_macro_tile_log = layerlog['tiles']['tile-0']['log']
        input_tiles_per_macro_tile = first_macro_tile_log['settings']['num-c-tiles']
        output_tiles_per_macro_tile = first_macro_tile_log['settings']['num-k-tiles']

        total_input_tiles = input_tiles_per_macro_tile * num_macro_tiles
        total_output_tiles = output_tiles_per_macro_tile * num_macro_tiles


        first_micro_tile = first_macro_tile_log['core']['tile-0']

        # tile sizes
        feature_tile_size = first_micro_tile['mems']['dat']['size']
        output_tile_size = first_micro_tile['mems']['out']['size']
        weight_tile_size = first_micro_tile['mems']['wt']['size']

        # single tile time
        tile_time = first_micro_tile['tile-time']
        
        # total layer time
        layer_time = layerlog['total-layertime']

        # collect all pieces of data
        df_rows.append({
            'Layer': f'{network_id}_{layer_name}',
            'Layer time': layer_time,
            'Single tile time': tile_time,
            'Input tiles': total_input_tiles,
            'Output tiles': total_output_tiles,
            'Input tile size': feature_tile_size,
            'Output tile size': output_tile_size,
            'Weight tile size': weight_tile_size,
        })
    
    # create complete dataframe and save
    final_df = pd.DataFrame(df_rows)
    final_df.to_excel(output_path, index=False)