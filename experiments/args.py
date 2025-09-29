from argparse import ArgumentParser

def parse_args() -> tuple[str, str]:
    parser = ArgumentParser(prog='run_experiment')

    parser.add_argument('experiment_folder', help='Path to the experiment folder.')
    parser.add_argument('configuration_name', help='Name of the YAML configuration file for the experiment. The file should be in the experiment folder.')

    args = parser.parse_args()
    return (args.experiment_folder, args.configuration_name)