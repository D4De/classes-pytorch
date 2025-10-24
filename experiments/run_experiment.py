import os
import csv
import yaml
import torch

from classes.simulators.pytorch.fault_list import PyTorchFaultListDynamic, PyTorchFaultListDynamicMetadata
from classes.simulators.pytorch.simulator_hook import create_simulator_hook
from classes.simulators.pytorch.error_model_mapper import create_module_to_generator_mapper_dynamic
from classes.simulators.pytorch.fault_list_datasets import FaultListFromTarFileDynamic

import experiments.logger as logger
import experiments.experiment_utils as utils

from experiments.args import parse_args
from experiments.network_getter import get_network_and_exp_functions


if __name__ == '__main__':
    #------------------------------------------------------------
    # initial setup

    experiment_dir, configuration_name, regenerate_faults = parse_args()

    # check if experiment directory exists
    try:
        utils.ensure_dir_exists_nonempty(experiment_dir)
    except Exception as e:
        print(e)
        raise ValueError(f'Experiment directory should exist and contain at least one configuration file.')

    # add extension to configuration file and check if it exists
    if not configuration_name.endswith('.yaml'):
        configuration_name = configuration_name + '.yaml'

    configuration_file_path = os.path.join(experiment_dir, configuration_name)

    if not os.path.exists(configuration_file_path):
        raise FileNotFoundError(f'Configuration file {configuration_file_path} does not exist.')

    # load configuration file
    with open(configuration_file_path, 'r') as config_file:
        config: dict = yaml.load(config_file, yaml.SafeLoader)

    # extract mandatory parameters from config
    key = 'experiment_name'
    try:
        experiment_name         = config[key]
        key = 'network_dataset_id'
        network_dataset_id      = config[key]
        key = 'batch_size'
        batch_size              = config[key]
        key = 'error_models_path'
        error_models_path       = os.path.realpath(config[key])
        key = 'num_faults_per_module'
        num_faults_per_module   = config[key]
        key = 'fault_list_path'
        fault_list_path         = os.path.realpath(config[key])
    except KeyError:
        raise ValueError(f'Required key {key} in experiment configuration file is missing.')

    # get optional parameters
    use_single_batch        = config.get('use_single_batch', True)
    tolerance               = config.get('tolerance', 0.001)
    num_threads             = config.get('num_threads', 4)
    compute_single_metrics  = config.get('compute_single_metrics', False)

    # ensure that error models path exists
    try:
        utils.ensure_dir_exists_nonempty(error_models_path, 'error_models_path')
    except Exception as e:
        print(e)
        raise FileNotFoundError(f'Error model directory {error_models_path} is empty')

    # prepare other directories if they don't exist
    outputs_dir = os.path.join(experiment_dir, 'outputs')
    logs_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # prepare logging
    current_datetime = utils.get_stringified_datetime()
    logfile_path = os.path.join(logs_dir, f'explog_{current_datetime}.txt')
    exp_logger = logger.create_experiment_logger(__name__, logfile_path=logfile_path)

    exp_logger.info(f'Started experiment - id is {network_dataset_id}')

    # paths for the overall report and the error report
    overall_report_path = os.path.join(outputs_dir, 'overall_report_' + current_datetime + '.yaml')
    error_report_path = os.path.join(outputs_dir, 'error_report_' + current_datetime + '.csv')

    # start collecting overall report data
    experiment_report_data = {}
    experiment_report_data['Experiment name'] = experiment_name
    experiment_report_data['Date and time'] = current_datetime

    #------------------------------------------------------------
    # get device
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'
    exp_logger.info(f'Using device {device} as accelerator')

    # get network, dataset and type of task
    exp_logger.info('Getting network, dataset and experiment functions')
    (
    model, dataloader, network_info,
    run_fn, metrics_fn
    ) = get_network_and_exp_functions(network_dataset_id, batch_size, device)
    

    num_samples = dataloader.batch_size if use_single_batch else len(dataloader) * dataloader.batch_size

    exp_logger.info(f'Loaded model and weights')
    exp_logger.info(f'Loaded dataset - batch size is {batch_size} - sample count is {num_samples}')

    model_name, dataset_name = network_dataset_id.split('_')

    experiment_report_data['Dataset data'] = {
        'Dataset name': dataset_name,
        'Number of samples': num_samples,
    }

    experiment_report_data['Model name'] = model_name
    experiment_report_data['Task'] = network_info.task
    experiment_report_data['Num. classes'] = network_info.num_classes

    # collect a few other hyperparameters
    experiment_report_data['Experiment hyperparameters'] = {
        'Batch size': batch_size,
        'Injected errors per module': num_faults_per_module,
        'Tolerance value for tensor equality': tolerance
    }
    #------------------------------------------------------------
    # AN OVERVIEW OF THE EXPERIMENT PROCESS
    # 1) The fault list for the current experiment is loaded or generated if not available.
    #
    # 2) A golden run with no layer corruption is performed. The results of this first run, i.e. the output tensors (or analogous classes) and
    # the corresponding labels/targets, are collected and saved to `golden_results`, which in general is a tuple that will be appropriately unpacked
    # by the functions associated to the network under test, as provided by `network_getter.py`.
    #
    # 3) A golden run postprocessing function is called. This function takes the intermediate outputs of step 1 and processes them in some way.
    # In the classification case, for example, the rankings corresponding to the golden scores are calculated, as well as metrics such as accuracy.
    # If some intermediate results need to be saved for later, the postprocessing function returns them, so that they can be packed together with
    # the previous results and used later. 
    #
    # 4) An error run is performed by injecting one fault at a time into the network and running the entire dataset through it.
    # The results of this run are saved as `error_results` and will again be unpacked as needed later.
    #
    # 5) An error run postprocessing function is called. The function takes the golden results and the error ones, computes the appropriate
    # metrics for each injection run and saves them, possibly along with other metadata. Regardless of the function implementation, it should
    # always return the number of masked, sdc safe and sdc critical results for an injection.

    # if the fault list does not exist or regeneration was requested, create it now
    if regenerate_faults or not os.path.exists(fault_list_path):
        exp_logger.info(f'Fault List {fault_list_path} does not exist in the current folder or regeneration was requested.' \
            f' Generating a new one with {num_faults_per_module} faults per network layer.')

        module_to_generator_mapping = create_module_to_generator_mapper_dynamic(
            model_folder_path=error_models_path,
            logger=exp_logger,
        )
        
        fault_list = PyTorchFaultListDynamic(
            model, 
            dataloader,
            module_filter_fn = lambda x: isinstance(x, torch.nn.Conv2d),
            module_to_fault_generator_fn=module_to_generator_mapping,
            device=device,
            logger=exp_logger,
        )
        fault_list.generate_and_persist_fault_list(fault_list_path, num_faults_per_module, logger=exp_logger, overwrite=True)
    
    # load the fault list metadata
    exp_logger.info('Loading fault list')
    fault_list_info = PyTorchFaultListDynamicMetadata.load_fault_list_info(fault_list_path)
    
    #------------------------------------------------------------
    # START ERROR INJECTION
    # these total metrics are meaningless in terms of individual SEUs, they are just for report completeness
    total_injected_errors = 0
    total_masked = 0
    total_sdc_safe = 0
    total_sdc_critical = 0

    layer_metrics_dict = {} # this will store the final metrics for each layer/fault injection pair

    # prepare csv log to store individual results for corrupted tensors
    error_report_csv = open(error_report_path, 'w', newline='')
    csv_writer = csv.writer(error_report_csv)
    csv_writer.writerow(network_info.csv_header)

    timer = utils.Timer()
    timer.start()

    for injectable_module_name in fault_list_info.injectable_layers:
        layer_metrics_dict[injectable_module_name] = {}
        module = model.get_submodule(injectable_module_name)

        # build a fault dataset for each module, if a compatible fault generator is available
        fault_list_dataset = FaultListFromTarFileDynamic(
            fault_list_path, injectable_module_name
        )
        fault_list_loader = torch.utils.data.DataLoader(
            fault_list_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )

        exp_logger.info(f'Starting injection in module {injectable_module_name} of type {type(module).__name__}')

        # iterate through the fault dataset and inject
        for fault_num, fault in enumerate(fault_list_loader):
            layer_metrics_dict[injectable_module_name][fault_num] = {}

            exp_logger.info(f'Error {fault_num} | Spatial Pattern: {fault.spatial_pattern_name}')

            fault.to(device=device)

            # create the injection hook
            error_simulator_pytorch_hook = create_simulator_hook(fault)

            # perform run
            results = run_fn(
                injected_module=module,
                error_simulator_pytorch_hook=error_simulator_pytorch_hook,
                use_single_batch=use_single_batch,
            )

            # compute metrics
            masked, sdc_safe, sdc_critical = metrics_fn(
                results=results,
                layer_metrics_dict=layer_metrics_dict,
                csv_writer=csv_writer,
                module_name=injectable_module_name,
                error_number=fault_num,
                fault=fault,
                tolerance=tolerance,
                outputs_path=outputs_dir,
                compute_single_metrics=compute_single_metrics,
                num_threads = num_threads,
            )

            # save fault results for this layer
            layer_metrics_dict[injectable_module_name][fault_num]['Masked'] = masked 
            layer_metrics_dict[injectable_module_name][fault_num]['SDC safe'] = sdc_safe
            layer_metrics_dict[injectable_module_name][fault_num]['SDC critical'] = sdc_critical
            layer_metrics_dict[injectable_module_name][fault_num]['Spatial class'] = str(fault.spatial_pattern_name)

            total_masked += masked
            total_sdc_safe += sdc_safe
            total_sdc_critical += sdc_critical
            

        total_injected_errors += num_faults_per_module * num_samples
    

    # injection done for all modules
    timer.stop()
    injection_runtime = timer.get_duration_as_str()

    exp_logger.info(f'Error simulation done - took {injection_runtime}')

    # finalize the csv report
    error_report_csv.close()

    # add overall report data and save
    experiment_report_data['Error simulation data'] = {
        'Total number of injected errors': total_injected_errors,
        'Total number of masked errors': total_masked,
        'Total number of safe SDCs': total_sdc_safe,
        'Total number of critical SDCs': total_sdc_critical,
        'Per-layer statistics': layer_metrics_dict,
        'Error simulation runtime': injection_runtime,
    }

    with open(overall_report_path, 'w') as overall_report_file:
        yaml.dump(experiment_report_data, overall_report_file, sort_keys=False)