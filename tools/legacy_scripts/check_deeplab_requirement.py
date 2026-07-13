"""
Examines the experiment networks. For each layer, it determines the corresponding error model or the models that need to be
interpolated. It signals what layers require a deeplab model.
"""
import os
import torch
import numpy as np
import pandas as pd

import experiments.network_getter as getter

from argparse import ArgumentParser

from classes.simulators.pytorch.module_profiler import module_shape_profiler

networks_and_input_sizes = [
    ('res50_cifar10', 32),
    ('alexnet_cifar10', 32),
    ('mobilenetv2_gtsrb', 32),
    ('deeplabv3_oxfordpet', 128),
]

hyperparameters = [
    'Channels_in',
    'Channels_out',
    'Input_size',
    'Kernel_size',
    'Padding',
]

num_neighbors = 5

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_df_path', help='Path to one of the error models\' df files. Specific configuration does not matter.')
    df_path = os.path.realpath(parser.parse_args().model_df_path)

    if not os.path.exists(df_path):
        raise FileNotFoundError(f'Error models\' df at {df_path} does not exist.')
    
    model_df = pd.read_excel(df_path, sheet_name=0, index_col=0, header=0)
    model_hyperparameters = model_df[hyperparameters]

    # normalize hyperparameters
    hyper_max = model_hyperparameters.max(axis=1)
    hyper_min = model_hyperparameters.min(axis=1)
    model_norm_hyper = model_hyperparameters.sub(hyper_min, axis=0).divide(hyper_max-hyper_min, axis=0)

    for network_name, input_size in networks_and_input_sizes:
        print(f'Checking network {network_name}')
        no_deeplab = True
        network = getter.get_network_and_exp_functions(network_name, 1, 'cpu', return_model_only=True)

        layer_names = []
        for layer_name, layer in network.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer_names.append(layer_name)

        dummy_input = torch.rand((1,3,input_size, input_size))
        layer_shapes = module_shape_profiler(network, dummy_input, layer_names, 'cpu')

        for layer_name in layer_names:
            layer: torch.nn.Conv2d = network.get_submodule(layer_name)
            layer_input_size: int = layer_shapes[layer_name][1][-1]

            layer_hyperparameters = pd.Series({
                'Channels_in' : layer.in_channels,
                'Channels_out': layer.out_channels,
                'Input_size'  : layer_input_size,
                'Kernel_size' : layer.kernel_size[-1],
                'Padding'     : layer.padding[-1],
            })

            # normalize layer hyperparameters
            hyper_max = layer_hyperparameters.max()
            hyper_min = layer_hyperparameters.min()
            layer_norm_hyper = (layer_hyperparameters - hyper_min) / (hyper_max - hyper_min)

            # find most appropriate model
            for model_name, model_hyper in model_norm_hyper[hyperparameters].iterrows():
                if np.allclose(layer_norm_hyper, model_hyper):
                    # found exact match
                    if 'deeplab' in model_name:
                        no_deeplab = False
                        print(f'Layer {layer_name}: selected exact model {model_name}, comes from DeepLab')
                    break
            else:
                # no match found. Interpolate
                distances: pd.DataFrame = ((model_norm_hyper - layer_norm_hyper) ** 2).sum(axis=1)
                min_distance_indices = distances.argsort()[:num_neighbors]

                closest_model_rows = model_norm_hyper.iloc[min_distance_indices]
                closest_model_names = closest_model_rows.index.tolist()

                # check if any of the closest models come from deeplab
                for closest_model_name in closest_model_names:
                    if 'deeplab' in closest_model_name:
                        no_deeplab = False
                        print(f'Layer {layer_name}: interpolation requires model {closest_model_name}, comes from DeepLab')
                        break

        if no_deeplab:
            print(f'Network {network_name} result: does NOT require DeepLab.\n')
        else:
            print(f'Network {network_name} result: REQUIRES DeepLab.\n')