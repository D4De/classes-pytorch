import re
import os
import yaml
import torch
import pandas as pd

import experiments.network_getter as netget

from argparse import ArgumentParser

initial_models_dirname               = 'initial_error_models'
merged_models_dirname                = 'merged_error_models'
netcontent_dirname                   = 'netcontent_files'
hw_unit_reports_dirname              = 'hw_unit_reports'
step1_output_filename                = 'step1_complete_df.xlsx'
step2_output_filename                = 'step2_unique_complete_df.xlsx'
step2_reconstruction_output_filename = 'step2_reconstruction_test_df.xlsx'

hyperparameters = [
    'Channels_in',
    'Channels_out',
    'Input_size',
    'Kernel_size',
    'Padding',
]

macroscopic_results = [
    'Masked',
    'SDC',
    'Crash+Hang',
]

spatial_classes = [
    'Single',
    'FullChannels',
	'Rectangles',
	'SingleChannelRandom',
	'ShatteredChannel',
	'QuasiShatteredChannel',
	'MultiChannelBlock',
	'BulletWake',
	'SameRow',
	'SingleBlock',
    'Skip4',
]

# not used in the actual postprocessing, but possibly identified by the classifier
extra_spatial_classes = [
    'SingleChannelAlternatedBlocks',
    'MultipleChannelsUncategorized',
]


def load_config_dict():
    parser = ArgumentParser()
    parser.add_argument('config_filepath', help='Path to the postprocess configuration yaml file.')
    args = parser.parse_args()

    config_path = os.path.realpath(args.config_filepath)
    if not os.path.exists(config_path):
        raise ValueError(f'Configuration file at {config_path} does not exist.')
    
    with open(config_path) as f:
        config_dict = yaml.load(f, yaml.SafeLoader)
    
    # check configuration consistency
    networks_and_layers: dict = config_dict['networks_and_layers']
    input_sizes: list = config_dict['network_input_sizes']
    if len(networks_and_layers) != len(input_sizes):
        raise ValueError(f'The dictionary of networks and the list of input sizes in the configuration file have different lengths.')

    return config_dict


def camelcase_to_snakecase(camel: str):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', camel).lower()


# network hyperparameters
def build_hyperparameters_dataframe(networks_and_layers: dict, input_sizes: list[int]):
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'

    network_dfs: list[pd.DataFrame] = []

    for network_entries, input_size in zip(networks_and_layers.items(), input_sizes):
        network_name, layers = network_entries
        # if network is already supplied by the getter, fetch it
        if network_name in netget.available:
            network = netget.get_network_and_exp_functions(network_name, 1, device, return_model_only=True)
        else:
            network = get_unlisted_network(network_name)

        network_dfs.append(get_network_hyperparameters(network, network_name, layers, input_size, device))
    
    return pd.concat(network_dfs)


def get_unlisted_network(network_name: str):
    """Network getter function for networks which are not covered by the standard network_getter."""

    match network_name:
        case 'lenet_cifar10':
            import nets_repo.classification.cifar10.models.lenet as lenet
            return lenet.LeNet()
        case 'vgg11_cifar10':
            import nets_repo.classification.cifar10.models.vgg_11 as vgg
            return vgg.VGG11CIFAR10()
        case 'res18_cifar100':
            import nets_repo.classification.cifar10.models.resnet as resnet
            return resnet.ResNet18()
        case _:
            raise ValueError(f'Network {network_name} is not supported.')


def get_network_hyperparameters(network: torch.nn.Module, network_name: str, layer_names: list[str], input_size: int, device):
    """Note: this function only works on convolutional layers. If other types of layers are passed in the names list, errors will
    likely be thrown."""
    
    df_rows: list[list] = []
    input_sizes: list[int] = []

    def _make_input_hook():
        def _hook(module, input, output):
            input_sizes.append(input[0].shape[-1])
            return output
        return _hook
    
    handles: list[torch.utils.hooks.RemovableHandle] = []

    # extract parameters and install hooks
    for layer_name in layer_names:
        layer = network.get_submodule(layer_name)
        df_rows.append([
            f'{network_name}_{layer_name}',
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size[0],
            layer.padding[0],
        ])

        handle = layer.register_forward_hook(_make_input_hook())
        handles.append(handle)

    # forward pass to get input sizes
    dummy_input = torch.rand((1,3,input_size, input_size))
    if device is not None:
        network = network.to(device)
        dummy_input = dummy_input.to(device)

    _ = network(dummy_input)
    for handle in handles:
        handle.remove()

    # add input_sizes to rows
    for row, input_size in zip(df_rows, input_sizes):
        row.insert(3, input_size)

    # build dataframe
    hyper_df = pd.DataFrame(df_rows, columns=['Layer','Channels_in','Channels_out','Input_size','Kernel_size','Padding'])
    return hyper_df.set_index('Layer')
