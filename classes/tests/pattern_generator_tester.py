import os
import traceback as tb

import argparse

from tqdm import tqdm

from classes.error_models.error_model import ErrorModel
from classes.error_models.error_model_entry import ErrorModelEntry
from classes.fault_generator.fault_generator import FaultGenerator
from classes.pattern_generators import get_default_generators

# You can change here the mapping with a custom one
GENERATOR_MAPPING = get_default_generators()

def test_error_models(
        folder_path : str, 
        shapes=[(1,1), (7,7), (16,16), (20,20), (32,32), (416,416), (128,64), (1,100), (100,1)], 
        n_channels=[1,3,4,12,32,100,128,137,256],
        n_iter=5,
        layout='CHW'):
    
    if os.path.isdir(folder_path):
        all_models_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    else:
        all_models_paths = [folder_path]
    total_progr = len(shapes) * len(n_channels) * len(all_models_paths)
    pbar = tqdm(total=total_progr)

    for file_path in all_models_paths:
        filename = os.path.basename(file_path)
        error_model = ErrorModel.from_json_file(file_path)
        f_gen = FaultGenerator(error_model, generator_mapping=GENERATOR_MAPPING)
        for shape in shapes:
            for n_channel in n_channels:
                pbar.set_description(f'{filename[:-5]} {n_channel}x{shape[0]}x{shape[1]}')
                if layout == 'CHW':
                    out_shape = (n_channel,) + tuple(shape)
                elif layout == 'HWC':
                    out_shape = tuple(shape) + (n_channel,)
                test_fault_generator(f_gen, out_shape, layout, pbar=pbar, n_attempts=n_iter)
                pbar.update(1)

def test_fault_generator(fault_gen : FaultGenerator, output_shape, layout = 'CHW', n_attempts = 100 , pbar : tqdm = None):
    for sp_name, gen_f, sp_params in fault_gen.spatial_patterns_generator():
        if pbar is not None:
            pbar.set_postfix_str(sp_name)
        flat_params = {**sp_params['keys'], **sp_params['stats']}
        for tr in range(n_attempts):
            try:
                is_random = any(
                    isinstance(val, dict) and "RANDOM" in val for val in flat_params.values()
                )
                if is_random:
                    realized_params = ErrorModelEntry.realize_random_parameters(flat_params)
                else:
                    realized_params = flat_params
                mask = gen_f(output_shape, realized_params, layout)
            except Exception as e:

                print(f'Failed generation')
                print(tb.format_exc())
                print(f'Attempt number: {tr}')
                print(f'Pattern: {sp_name}')
                print(f'Shape: {output_shape}')
                print(f'Layout: {layout}')
                print(f'Original params: {sp_params}')
                print(f'Realized Params: {realized_params}')
                exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        "Pattern Generator Tester",
        description="""
            Tool for testing the correctness of the error model pattern generators.

            The tool loads one or more error model in the classes .json format and tries to generate each configuration
            of spatial class and spatial parameters in the error models. The tool can also visualize the generated patterns
            helping the developer to check if the patterns are generated correctly.
        """
    )

    parser.add_argument(
        "error_model_path", type=str, help="Path to a folder of .json error models or to a single .json error model file." 
    )

    parser.add_argument(
        "-l", "--layout", default="CHW", type=str, help="Order of the tensors dimensions. For now CHW (PyTorch ordering) and HWC (TensorFlow ordering) are supported"
    )

    parser.add_argument(
        "-i", "--iterations", default=5, type=int, help="Number of masks generated for each possible configuration of spatial parameters. Default is 5."
    )

    parser.add_argument(
        "-v", "--visualize", action='store_true', help="Generates visualization for each generated mask, saving them to image files."
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    test_error_models(args.error_model_path, n_iter=args.iterations, layout=args.layout)
