import os
import yaml
import experiments.console_utils as console_utils

if __name__ == '__main__':
    print(f'{'*'*5} CLASSES experiment configuration builder {'*'*5}')
    print('--NOTE: you may use both absolute and relative paths--\n')

    # this_dir = os.path.dirname(os.path.realpath(__file__))

    # # configuration output path
    # output_dir = os.path.join(this_dir, 'experiment_configurations')
    # if console_utils.yes_no_choice_interactive(
    #     f'The configuration file will be saved in {output_dir}. Would you like to choose a different directory?'
    # ):
    #     output_dir = os.path.realpath(
    #         console_utils.read_string_non_empty_interactive('Enter the path of the directory to save the new configuration to. Will be created if it does not exist.')
    #     )

    # os.makedirs(output_dir, exist_ok=True)
    # console_utils.print_console_separator()
    # #-------------------------------------------------------------------------------------------------


    # # configuration file name
    # output_name = console_utils.read_string_non_empty_interactive('Enter the name of the configuration file you\'re creating.')
    # output_path = os.path.join(output_dir, output_name + '.yaml')
    # console_utils.print_console_separator()
    # #-------------------------------------------------------------------------------------------------


    # experiment directory and name
    experiment_dir = os.path.realpath(
        console_utils.read_string_non_empty_interactive(
            'Enter the path to the directory that will be used for the experiment.\n' \
            'The configuration file will be saved in it as \'exp_config.yaml\'. The directory does not need to exist.'
        )
    )
    configuration_file_path = os.path.join(experiment_dir, 'exp_config.yaml')
    os.makedirs(experiment_dir, exist_ok=True)
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------

    experiment_name = os.path.basename(experiment_dir)
    if console_utils.yes_no_choice_interactive(
        f'The experiment will be named {experiment_name}. Would you like to choose a different name?'
    ):
        experiment_name = console_utils.read_string_non_empty_interactive('Enter experiment name.')
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------


    # network model
    model_name = console_utils.read_string_non_empty_interactive('Enter the name of the network model under test. Ensure you use a valid id (check list in `network_getter.py` if necessary)')
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------

    # dataset
    dataset_name = console_utils.read_string_non_empty_interactive('Enter the name of the dataset for the experiment. Ensure you use a valid id (check list in `network_getter.py` if necessary)')
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------


    batch_size = console_utils.read_number_interactive(
        'Enter batch size to use for the experiment.',
        type=int,
        force_positive=True
    )
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------


    # error models
    error_models_path = console_utils.read_existing_path_interactive(
        'Enter the path to the directory containing the error models you want to use.',
        must_be_dir=True
    )
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------


    # tolerance
    tolerance = 1e-3
    if console_utils.yes_no_choice_interactive(
        'The default tolerance value to check for tensor equality is 1e-3. Would you like to choose a different one?'
    ):
        tolerance = console_utils.read_number_interactive(
            'Enter tolerance value to use for the experiments.',
            type=float,
            force_positive=True
        )
    
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------


    # number of faults
    num_faults = console_utils.read_number_interactive(
        'Enter the number of faults to inject PER MODULE.',
        type=int,
        force_positive=True
    )
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------


    # fault list
    fault_list_name = '{}_{}_fault_list.tar'.format(model_name, dataset_name)
    if console_utils.yes_no_choice_interactive(
        'The fault list for the experiment will be generated in the experiment directory.\n' \
        f'The default name for the generated fault list is {fault_list_name}. Would you like to change it?'
    ):
        fault_list_name = console_utils.read_string_non_empty_interactive('Enter the name of the fault list to generate.')
        # add extension if missing
        if not fault_list_name.endswith('.tar'):
            fault_list_name = fault_list_name + '.tar'

    fault_list_path = os.path.join(experiment_dir, fault_list_name)
    console_utils.print_console_separator()
    #-------------------------------------------------------------------------------------------------


    # build final dictionary
    experiment_dict = {
        'experiment_name':          experiment_name,
        'network_dataset_id':       '{}_{}'.format(model_name, dataset_name),
        'batch_size':               batch_size,
        'error_models_path':        error_models_path,
        'tolerance':                tolerance,
        'num_faults_per_module':    num_faults,
        'fault_list_path' :         fault_list_path
    }

    with open(configuration_file_path, 'w') as f:
        yaml.dump(experiment_dict, f, sort_keys=False)

    print(f'ALL DONE: your configuration file is {configuration_file_path}.')