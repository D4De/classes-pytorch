import os
import yaml

class Args:
    error_models_base_path      :str
    experiments_base_path       :str
    final_reports_base_path     :str
    output_dir                  :str
    network_dataset_ids         :list[str]
    configuration_ids           :list[str]
    short_configuration_ids     :list[str]

    def __init__(self, args_file_path: str):
        args_file_path = os.path.realpath(args_file_path)
        if not os.path.exists(args_file_path):
            raise FileNotFoundError(f'Args file at {args_file_path} does not exist.')
        
        with open(args_file_path) as f:
            args_dict = yaml.load(f, yaml.SafeLoader)
        
        self.error_models_base_path     = os.path.realpath(args_dict['error_models_base_path'])
        self.experiments_base_path      = os.path.realpath(args_dict['experiments_base_path'])
        self.final_reports_base_path    = os.path.realpath(args_dict['final_reports_base_path'])
        self.output_dir                 = os.path.realpath(args_dict['output_dir'])
        self.network_dataset_ids        = args_dict['network_dataset_ids']
        self.configuration_ids          = args_dict['configuration_ids']
        self.short_configuration_ids    = args_dict['short_configuration_ids']

        # shallow existence checks
        if not os.path.isdir(self.error_models_base_path):
            raise ValueError(f'Error models directory at {self.error_models_base_path} does not exist.')
        if not os.path.isdir(self.experiments_base_path):
            raise ValueError(f'CLASSES experiments directory at {self.experiments_base_path} does not exist.')
        if not os.path.isdir(self.final_reports_base_path):
            raise ValueError(f'Path to final yaml network reports at {self.final_reports_base_path} is not a directory.')
