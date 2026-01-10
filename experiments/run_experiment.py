import os
import csv
import yaml
import torch
import tarfile

from classes.simulators.pytorch.fault_list import PyTorchFaultListDynamic, PyTorchFaultListDynamicMetadata
from classes.simulators.pytorch.simulator_hook import create_simulator_hook
from classes.simulators.pytorch.module_profiler import module_range_profiler, module_shape_profiler
from classes.simulators.pytorch.error_model_mapper import load_error_models, map_layers_to_error_models
from classes.simulators.pytorch.fault_list_datasets import FaultListFromTarFileDynamic

import experiments.logger as logger
import experiments.experiment_utils as utils

from experiments.args import parse_args
from experiments.network_getter import get_network_and_exp_functions, requires_single_metrics
from experiments.results_to_dictionary import pack_report_dictionary


def main():
    #------------------------------------------------------------
    # INITIAL SETUP

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
        experiment_name: str          = config[key]

        key = 'network_dataset_id'
        network_dataset_id: str       = config[key]

        key = 'hw_config_id'
        hw_config_id: str             = config[key]

        key = 'error_models_path'
        error_models_path: str        = os.path.realpath(config[key])

        key = 'error_models_df_path'
        error_models_df_path: str     = os.path.realpath(config[key])

        key = 'batch_size'
        batch_size: int               = config[key]

        key = 'uniform_spatial_classes'
        uniform_spatial_classes: bool = config[key]

        key = 'num_faults_per_module'
        num_faults_per_module: int    = config[key]

        key = 'fault_list_path'
        fault_list_path: str          = os.path.realpath(config[key])

        key = 'SDC_frequencies_path'
        SDC_frequencies_path: str = os.path.realpath(config[key])
    except KeyError:
        raise ValueError(f'Required key {key} in experiment configuration file is missing.')

    # get optional parameters
    use_single_batch        = config.get('use_single_batch', True)
    tolerance               = config.get('tolerance', 0.001)
    num_threads             = config.get('num_threads', 4)
    compute_single_metrics  = config.get('compute_single_metrics', False)

    # if the network explicitly requires the calculation of single metrics (e.g. YOLO), override the setting
    compute_single_metrics = requires_single_metrics(network_dataset_id)

    # ensure that error models path exists
    try:
        utils.ensure_dir_exists_nonempty(error_models_path, 'error_models_path')
    except Exception as e:
        print(e)
        raise FileNotFoundError(f'Error model directory {error_models_path} is empty')

    # prepare other directories if they don't exist
    outputs_dir = os.path.join(experiment_dir, 'outputs')
    logs_dir    = os.path.join(experiment_dir, 'logs')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # prepare logging
    logfile_path = os.path.join(logs_dir, f'explog_{hw_config_id}_{num_faults_per_module}err_{batch_size}in.txt')
    exp_logger   = logger.create_experiment_logger(__name__, logfile_path=logfile_path)

    exp_logger.info(f'Started experiment - id is {network_dataset_id}')

    # paths for the overall report and the error report
    overall_report_path = os.path.join(outputs_dir, f'overall_report_{hw_config_id}_{num_faults_per_module}err_{batch_size}in.yaml')
    error_report_path   = os.path.join(outputs_dir, f'error_report_{hw_config_id}_{num_faults_per_module}err_{batch_size}in.csv')

    # start collecting overall report data
    experiment_report_data = {}
    experiment_report_data['Experiment name'] = experiment_name
    experiment_report_data['Date and time']   = utils.get_stringified_datetime()

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
        'Dataset name'     : dataset_name,
        'Number of samples': num_samples,
    }

    experiment_report_data['Model name']   = model_name
    experiment_report_data['Task']         = network_info.task
    experiment_report_data['Num. classes'] = network_info.num_classes

    # collect a few other hyperparameters
    experiment_report_data['Experiment hyperparameters'] = {
        'Batch size'                         : batch_size,
        'Injected errors per module'         : num_faults_per_module,
        'Tolerance value for tensor equality': tolerance
    }
    #------------------------------------------------------------
    # AN OVERVIEW OF THE EXPERIMENT PROCESS
    # 1) The fault list for the current experiment is loaded or generated if not available. At the same time, the SDC frequencies for
    # the layers are computed during fault generation and saved to a file.
    #
    # 2) For each generated fault, the experiment run function is called; this function executes a golden run and the error run.
    # The results are collected and returned.
    #
    # 3) The results are passed to the metrics calculation function, which determines which runs resulted in masking and which led
    # to SDC (safe or critical). Depending on the configuration settings, metrics on the SDC results are directly calculated or the
    # results are saved to file for later postprocessing.
    # 
    # 4) The overall results, both for the whole experiment and for the single layers, are gathered and saved to a final report. 
    # The report's results for the single spatial classes for each layer are also computed and saved to a separate report.


    #------------------------------------------------------------
    # FAULT LIST GENERATION
    # if the fault list does not exist or regeneration was requested, create it now
    if regenerate_faults or not os.path.exists(fault_list_path):
        exp_logger.info(f'Fault List {fault_list_path} does not exist in the current folder or regeneration was requested.' \
            f' Generating a new one with {num_faults_per_module} faults per network layer.')

        # step 1: define a filter function to extract injectable layers from the network and get those layers' names
        layer_filter_fn = lambda x: isinstance(x, torch.nn.Conv2d)
        layer_names: list[str] = []
        for name, layer in model.named_modules():
            if layer_filter_fn(layer):
                layer_names.append(name)
        exp_logger.info(f'Found a total of {len(layer_names)} injectable layers in the network')


        # step 2: execute network profilation to obtain the layers' parameters
        exp_logger.info('Starting network profilation')
        for input_sample, _ in dataloader: break    # get an input sample for profilation
        layer_shapes = module_shape_profiler(model, input_sample, layer_names, device)
        layer_ranges = module_range_profiler(model, dataloader, layer_names, device)
        # sanity check: the two dictionaries should have the same keys
        if not layer_shapes.keys() == layer_ranges.keys():
            raise KeyError(f'The keys for layer shapes and layer ranges do not match: {layer_shapes.keys()=}\n{layer_ranges.keys()=}')


        # step 3: load error models and their parameters
        exp_logger.info('Loading error models')
        error_model_df, error_model_dicts = load_error_models(error_models_path, error_models_df_path)


        # step 4: match each layer to its error model (existing or interpolated)
        exp_logger.info('Mapping layers to error models')
        layer_sdc_frequencies, layer_fault_generators = map_layers_to_error_models(
            model, layer_names, layer_shapes, error_model_df, error_model_dicts, exp_logger
        )
        

        # step 5: save SDC frequency dictionary for later postprocessing
        exp_logger.info('Saving SDC and class frequencies file.')
        if not SDC_frequencies_path.endswith('.xlsx'):
            SDC_frequencies_path = SDC_frequencies_path + '.xlsx'
        layer_sdc_frequencies.to_excel(SDC_frequencies_path, sheet_name='Frequencies', index_label='Layer')


        # step 6: generate fault list       
        fault_list = PyTorchFaultListDynamic(
            layer_names,
            layer_shapes,
            layer_ranges,
            layer_fault_generators,
            input_sample,
            logger=exp_logger,
        )
        fault_list.generate_and_persist_fault_list(
            fault_list_path,
            num_faults_per_module,
            uniform_spatial_classes,
            logger=exp_logger,
            overwrite=True
        )
    

    #------------------------------------------------------------
    # START ERROR INJECTION
    # these total metrics are meaningless in terms of individual SEUs, they are just for report completeness
    total_masked = total_sdc_safe = total_sdc_critical = 0

    layer_metrics_dict = {} # this will store the final metrics for each layer/fault injection pair

    if compute_single_metrics:
        # prepare csv log to store individual results for corrupted tensors
        error_report_csv = open(error_report_path, 'w', newline='')
        csv_writer = csv.writer(error_report_csv)
        csv_writer.writerow(network_info.csv_header)
    else:
        error_report_csv = csv_writer = None


    timer = utils.Timer()
    timer.start()


    # load the fault list metadata and get relevant info
    exp_logger.info('Loading relevant metadata from fault list')
    fault_list_info = PyTorchFaultListDynamicMetadata.load_fault_list_info(fault_list_path)
    if not 'layer_names' in locals():
        layer_names = fault_list_info.injectable_layers
    num_module_faults = fault_list_info.num_module_faults

    # open the tar fault list and start injecting
    with tarfile.TarFile(fault_list_path, "r") as tarf:

        for injectable_module_name, module_num_faults in zip(layer_names, num_module_faults):
            layer_metrics_dict[injectable_module_name] = {}
            module = model.get_submodule(injectable_module_name)

            # build a fault dataset for each module
            fault_list_dataset = FaultListFromTarFileDynamic(
                tarf,
                injectable_module_name,
                module_num_faults,
                fault_list_info.fault_batch_size,
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
                    injected_module              = module,
                    error_simulator_pytorch_hook = error_simulator_pytorch_hook,
                    use_single_batch             = use_single_batch,
                )

                # compute metrics
                masked, sdc_safe, sdc_critical = metrics_fn(
                    results                = results,
                    layer_metrics_dict     = layer_metrics_dict,
                    csv_writer             = csv_writer,
                    module_name            = injectable_module_name,
                    error_number           = fault_num,
                    fault                  = fault,
                    tolerance              = tolerance,
                    outputs_path           = outputs_dir,
                    compute_single_metrics = compute_single_metrics,
                    num_threads            = num_threads,
                )

                # save fault results for this layer
                layer_metrics_dict[injectable_module_name][fault_num]['Masked']        = masked 
                layer_metrics_dict[injectable_module_name][fault_num]['SDC safe']      = sdc_safe
                layer_metrics_dict[injectable_module_name][fault_num]['SDC critical']  = sdc_critical
                layer_metrics_dict[injectable_module_name][fault_num]['Spatial class'] = str(fault.spatial_pattern_name)

                total_masked       += masked
                total_sdc_safe     += sdc_safe
                total_sdc_critical += sdc_critical
        

    # injection done for all modules
    timer.stop()
    injection_runtime = timer.get_duration_as_str()

    exp_logger.info(f'Error simulation done - took {injection_runtime}')

    # finalize the csv report
    if error_report_csv is not None:
        error_report_csv.close()

    # add overall report data and save
    experiment_report_data['Error simulation data'] = {
        'Total number of injected errors' : total_masked + total_sdc_safe + total_sdc_critical,
        'Total number of masked errors'   : total_masked,
        'Total number of safe SDCs'       : total_sdc_safe,
        'Total number of critical SDCs'   : total_sdc_critical,
        'Per-layer statistics'            : layer_metrics_dict,
        'Error simulation runtime'        : injection_runtime,
    }

    exp_logger.info('Saving final reports')

    with open(overall_report_path, 'w') as overall_report_file:
        yaml.dump(experiment_report_data, overall_report_file, sort_keys=False)

    # compute the total result frequencies (also divided by spatial class) and produce a second output file
    aggregated_report = pack_report_dictionary(experiment_report_data)
    aggregated_report_filename = f'applev_{model_name}_{dataset_name}_{hw_config_id}_{num_samples}in_{num_faults_per_module}err.yaml'
    aggregated_report_filepath = os.path.join(experiment_dir, aggregated_report_filename)
    with open(aggregated_report_filepath, 'w') as aggregated_report_file:
        yaml.dump(aggregated_report, aggregated_report_file, sort_keys=False)


if __name__ == '__main__':
    main()