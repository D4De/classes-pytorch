from argparse import ArgumentParser

def parse_args() -> tuple[str, str]:
    parser = ArgumentParser(prog='run_experiment')

    parser.add_argument('experiment_folder', help='Path to the experiment folder.')
    parser.add_argument('-cf', '--configuration_name', help='Name of the YAML configuration file for the experiment. The file should be in the experiment folder.',
                        default='exp_config.yaml')
    parser.add_argument('-rf', '--regenerate_faults', help='If set, the fault list for the experiment is forcibly regenerated, overwriting the previous one.',
                        action='store_true')

    args = parser.parse_args()
    return (args.experiment_folder, args.configuration_name, args.regenerate_faults)