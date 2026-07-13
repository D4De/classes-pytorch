import os
import yaml
import pandas as pd

experiments_dir = '/home/miele/WORKSPACE/results-storage/error_simulation/experiments'
outfile = './error_counts.csv' # output file path

networks = [    # networks to consider; these are the same names used for the exp_* directories
    'alexnet_cifar10',
    'deeplabv3_oxfordpet',
    'res9_cifar10',
    'res50_cifar10',
    'yolov11_coco',
    'mobilenetv2-large_gtsrb',
]
bitwidths = ['int8', 'int16', 'int32']

if __name__ == '__main__':
    count_df = pd.DataFrame(0, index=networks, columns=bitwidths)

    for network in networks:
        network_dir = os.path.join(experiments_dir, f'exp_{network}')
        # iterate through subdirectories
        subdirs = [os.path.join(network_dir, subdir) for subdir in os.listdir(network_dir) if os.path.isdir(os.path.join(network_dir, subdir))]
        for subdir in subdirs:
            # look for an applev file
            applev_name = None
            for filename in os.listdir(subdir):
                if filename.startswith('applev'):
                    applev_name = filename
                    break
            if applev_name is None:
                continue

            # extract bitwidth and number of errors from the file name
            pieces = applev_name.split('_')
            bitwidth = pieces[-3]
            errors = int(pieces[-1][:-8])

            applev_path = os.path.join(network_dir, applev_name)
            with open(applev_path) as f:
                applev_dict = yaml.load(f, yaml.SafeLoader)
            
            applev_count = 0

            for layer, layer_dict in applev_dict.items():
                for spatial_class, class_dict in layer_dict.items():
                    # skip if not spatial class entry
                    if spatial_class.startswith('prob'):
                        continue
                    # skip if all fields are 0
                    if class_dict['masked'] == 0.0 and class_dict['sdc_safe'] == 0.0 and class_dict['sdc_critical'] == 0.0:
                        continue
                    
                    applev_count += errors
            
            # add total to dataframe
            count_df.at[network, bitwidth] += applev_count
    
    count_df.to_csv(outfile)