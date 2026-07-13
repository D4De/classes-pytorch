import torch
import torch.nn.modules as modtypes
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
import yaml

from console_utils import print_console_separator, select_index_from_list_interactive, read_separated_string_list_interactive

def generate_model_graph(model: torch.nn.Module, input_size: torch.Size, tensorboard_dir: str):
    """
    Given a network, generates the corresponding TensorBoard graph. Generation is skipped if
    the graph file already exists. Parameter 'input_size' can in principle be an arbitrary 4D tensor,
    but it's better to use the size of actual tensors that will be fed to the network, since the graph
    will also report all intermediate tensor sizes.
    """
    if len(input_size) != 4:
        raise ValueError(f'Model graph generation: input_size must be 4D, was {len(input_size)}D.')
    
    if not os.path.isdir(tensorboard_dir):
        raise ValueError(f'Model graph generation: provided path {tensorboard_dir} is not a directory.')
    
    save_dir = os.path.join(tensorboard_dir, 'graph_visual')
    if os.path.isdir(save_dir) and os.listdir(tensorboard_dir):
            print(f'--Model graph generation: graph directory "{save_dir}" is not empty. Skipping graph generation.')
    else:
        writer = SummaryWriter(save_dir)
        mock_input = torch.randn(input_size)
        writer.add_graph(model, mock_input)
        writer.close()
        print('--Model graph generation: graph generated and saved.')
    
    print('If TensorBoard is not already open, run')
    print(f'\t tensorboard --logdir={tensorboard_dir}')
    print('in a shell and open a browser tab as indicated (usually at http://localhost:6006/).')
    print('Otherwise, refresh the TensorBoard browser tab to see the graph.')
    print_console_separator()
    
        
def get_submodules(model: torch.nn.Module):
    """
    Given a network, returns a dictionary {module_name: module}, containing all
    network submodules with their fully qualified names (last nesting level, no module with children).
    """
    extracted_submodules = {}
    
    def _no_children(m: torch.nn.Module):
        children = m.named_children()
        # if the iterator is done, there are no children
        try:
            next(children)
        except StopIteration:
            return True
        return False
    
    # filter out all modules with children
    for modname, mod in model.named_modules():
        if _no_children(mod):
            extracted_submodules[modname] = mod
    
    return extracted_submodules


def group_conv_modules_by_kernel_size(modules: dict[str, torch.nn.Module], output_file: str):
    """
    Given a dictionary of network modules, it groups together the convolutional ones with the same
    kernel size and writes the result in a file. All non-conv modules are ignored.
    """
    conv_types = {}

    for modname in modules:
        mod = modules[modname]
        if type(mod) == modtypes.Conv2d:
            kernel_size = mod.kernel_size
            
            if type(kernel_size) == int:
                kernel_size = (kernel_size, kernel_size)
            
            conv_label = f'conv{kernel_size[0]}x{kernel_size[1]}'
            if conv_label not in conv_types:
                # create new list of module names for this kernel size
                conv_types[conv_label] = [modname]
            else:
                # add this module name to the existing list
                conv_types[conv_label] += [modname]
    
    with open(output_file, 'w') as f:
        yaml.dump(conv_types, f)


def list_submodules_with_filters(modules: dict[str, torch.nn.Module]):
    """
    Given a dictionary of network modules, prints them for inspection. The
    user can choose one of the provided filters to only show the modules of that type.
    """
    filters = {
        "conv":         [modtypes.Conv1d, modtypes.Conv2d, modtypes.Conv3d],
        "batchnorm":    [modtypes.BatchNorm1d, modtypes.BatchNorm2d, modtypes.BatchNorm3d],
        "relu":         [modtypes.ReLU],
        "maxpool":      [modtypes.MaxPool1d, modtypes.MaxPool2d, modtypes.MaxPool3d],
        "avgpool":      [modtypes.AvgPool1d, modtypes.AvgPool2d, modtypes.AvgPool3d],
        "linear":       [modtypes.Linear]
    }

    filter_choices = ['all'] + list(filters.keys())
    
    while True:
        selected_index = select_index_from_list_interactive(
            choices=filter_choices, 
            prompt_msg='--Module selection: what network modules would you like to list?',
            allow_none=True
        )
        
        if selected_index == None:
            break
        elif selected_index == 0:
            # 'all' -> show every module
            for modname in modules:
                print(f'{modname:<25} --> {modules[modname]}')
        else:
            selected_filters = filters[filter_choices[selected_index]]
            layer_filter = lambda m: (type(m) in selected_filters)
            for modname in modules.keys():
                if layer_filter(modules[modname]):
                    print(f'{modname:<25} --> {modules[modname]}')
        
    print_console_separator()


def select_submodules(modules: dict[str, torch.nn.Module], selection_list_dir: str):
    """
    Given a dictionary containing all network modules, returns a subdictionary
    containing only the selected modules. First scans the provided list directory,
    looking for '.modlist' files and, if found, asks the user to choose one.
    If no file exists, asks the user to select the modules they want and saves the new
    list as 'date_time.modlist' to the provided directory.
    """
    selected_modules = {}
    
    if not os.path.isdir(selection_list_dir):
        raise ValueError(f'Module selection: provided path {selection_list_dir} is not a directory.')
    
    # scan directory and look for '.modlist' files
    list_files = [file
                  for file in os.listdir(selection_list_dir) 
                  if file.endswith('.modlist')]
    
    if list_files:
        # found something: list the files and choose one
        selected_index = select_index_from_list_interactive(
            choices=list_files,
            prompt_msg='--Module selection: found the following module list files. Choose one or enter \'none\' to make a new selection.',
            allow_none=True
        )
        
        if selected_index is not None:
            # read from the selected file and get the modules
            selected_file_path = os.path.join(selection_list_dir, list_files[selected_index])
            with open(selected_file_path, 'r') as f:
                for modname in f.readlines():
                    modname = modname.strip().lower()
                    selected_modules[modname] = modules[modname]
            
            print_console_separator()
            return selected_modules
    else:
        # no saved list: ask the user
        print(f'--Module selection: no module list file found in {selection_list_dir}: making new selection.')
    
    # make new selection and save to file
    list_file_name = datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S') + '.modlist'
    list_file_path = os.path.join(selection_list_dir, list_file_name)

    with open(list_file_path, 'w') as f:
        while True:
            response = read_separated_string_list_interactive(
                prompt_msg=
                    '--Module selection: enter the modules you want to select, using their fully qualified names, comma-separated if more than one at a time.' \
                    'Unrecognized/already selected modules will be skipped. Enter \'none\' when you\'re done.',
                separator=','
            )

            if response[0] == 'none':
                if len(selected_modules) == 0:
                    print('--Module selection: you must select at least one module before proceeding.\n')
                    continue
                break
            
            for name in response:
                if name not in modules or name in selected_modules:
                    print(f'Module {name} does not exist or was already selected. Skipping.')
                else:
                    # add module name to list file
                    f.write(name + '\n')
                    selected_modules[name] = modules[name]
            print('\n')
            
    print(f'--Module selection: saved list to {list_file_path}')
    
    print_console_separator()
    return selected_modules