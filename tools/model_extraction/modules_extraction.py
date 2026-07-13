import torch
from torch.utils.hooks import RemovableHandle
import os
import numpy as np
import yaml

def generate_random_input(size: torch.Size):
    num_dimensions = len(size)
    
    if num_dimensions > 4 or num_dimensions < 2:
        raise ValueError(f'Random input generation: given size is {num_dimensions}D, must be at least 2D, at most 4D')
    
    if num_dimensions == 2:
        print('--Random input generation: given size is 2D, assuming value 1 for missing dimensions (batch, channels).')
        return torch.randn(1, 1, size[0], size[1])
    elif num_dimensions == 3:
        print('--Random input generation: given size is 3D, assuming value 1 for missing dimension (batch).')
        return torch.randn(1, size[0], size[1], size[2])
    
    return torch.randn(size)


def create_modules_dirs(selected_modules: dict[str, torch.nn.Module], modules_data_dir: str):
    """
    Creates the directory tree to save the data for the selected modules. If the provided directory
    does not exist, it's created as well. Each subdirectory is named after its corresponding module.
    """
    if os.path.exists(modules_data_dir) and not os.path.isdir(modules_data_dir):
        raise ValueError(f'Module data directory creation: provided path to save module data {modules_data_dir} is not a directory')
    
    if not os.path.exists(modules_data_dir):
        os.mkdir(modules_data_dir)
    
    for modname in selected_modules:
        module_dir = os.path.join(modules_data_dir, modname)
        
        if not os.path.exists(module_dir):
            os.mkdir(module_dir)


def save_selected_modules_parameters(selected_modules: dict[str, torch.nn.Module], modules_data_dir: str):
    """
    Populates the indicated directory tree with the parameters of the provided modules.
    Note: it currently works only for convolutions. For each conv module, it saves its weight and bias as
    numpy files and other relevant parameters to a yaml file.
    """
    if os.path.exists(modules_data_dir) and not os.path.isdir(modules_data_dir):
        raise ValueError(f'Module parameter saving: provided path to save module data {modules_data_dir} is not a directory')
    
    for modname in selected_modules:
        params = {}
        params['name'] = modname
        params['inputs'] = {}

        module_dir = os.path.join(modules_data_dir, modname)
        mod = selected_modules[modname]
        
        if 'conv' in str(type(mod)).lower():
            params['optype'] = 'conv'
            params['params'] = {
                'dilation-x': mod.dilation[0],
                'dilation-y': mod.dilation[1],
                'padding-x': mod.padding[0],
                'padding-y': mod.padding[1],
                'stride-x': mod.stride[0],
                'stride-y': mod.stride[1]
            }
            # save path to features file
            features_file = os.path.join(module_dir, 'features.npy')
            params['inputs']['features'] = features_file

            # save weights file and path to it
            weight_file = os.path.join(module_dir, 'weights')
            weight = mod.weight.detach().numpy()
            np.save(weight_file, weight)
            params['inputs']['weights'] = weight_file + '.npy'
            
            # if present, save bias file and path to it
            if mod.bias is not None:
                bias_file = os.path.join(module_dir, 'bias')
                bias = mod.bias.detach().numpy()
                np.save(bias_file, bias)
                params['inputs']['bias'] = bias_file + '.npy'

            # save parameter file
            params_file = os.path.join(module_dir, f'{modname}.yaml')
            with open(params_file, 'w') as f:
                yaml.dump(params, f)
        else:
            print(f'--Module parameters saving: {modname}: no procedure defined for this module. Skipping')
        # TODO: add support for Linear layers
            

def install_extraction_hooks(selected_modules: dict[str, torch.nn.Module], modules_data_dir:str) -> list[RemovableHandle]:
    """
    Installs a hook on each module in the provided dictionary. When a module's forward() method is run during inference,
    the hook takes its input tensor and saves it to a numpy file in the module's directory in the provided directory tree.
    """
    if os.path.exists(modules_data_dir) and not os.path.isdir(modules_data_dir):
        raise ValueError(f'Extraction hook installation: provided path to save module data {modules_data_dir} is not a directory')
    
    def _make_extraction_hook(module_name):
        input_file = os.path.join(modules_data_dir, module_name, 'features')
        
        def _extraction_hook(module, input, output):
            input_np = input[0].cpu().detach().numpy()
            np.save(input_file, input_np)
            return output
        
        return _extraction_hook
    
    handles: list[RemovableHandle] = []
    
    for modname in selected_modules:
        mod = selected_modules[modname]
        handle = mod.register_forward_hook(_make_extraction_hook(modname))
        handles.append(handle)
    
    return handles


def remove_extraction_hooks(handles: list[RemovableHandle]):
    for handle in handles:
        handle.remove()