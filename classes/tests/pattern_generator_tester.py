import os
import traceback as tb

import argparse
from typing import Literal, Optional, Sequence, Tuple

from tqdm import tqdm

from classes.error_models.error_model import ErrorModel
from classes.error_models.error_model_entry import ErrorModelEntry
from classes.fault_generator.fault_generator import FaultGenerator
from classes.pattern_generators import get_default_generators
from classes.visualization.mask import plot_mask

# You can change here the mapping with a custom one
GENERATOR_MAPPING = get_default_generators()


def test_error_models(
    models_path: str,
    output_path : Optional[str],
    channel_sizes : Sequence[Tuple[int,int]]=[
        (1, 1),
        (7, 7),
        (16, 16),
        (20, 20),
        (32, 32),
        (416, 416),
        (128, 64),
        (1, 100),
        (100, 1),
    ],
    n_channels : Sequence[int] = [1, 3, 4, 12, 32, 100, 128, 137, 256],
    n_iter=5,
    layout : Literal['CHW','HWC'] ="CHW",
):
    """
    Run an extensive test that explore extensively the parameter space generated by the error models.
    This can be used to test that the error models work properly before launching extensive error simulation campagins.

    Args
    ---
    * ``models_path : str``. A path to a directory of models or to a single .json error model file. If the path is a directory all the models inside it will be tested.
    * ``output_path : Optional[str]``. Path to a folder that will contain the images generated. If not specified the images will not be generated.
    * ``channel_sizes : Sequence[Tuple[int,int]]``.The sizes of the channels to be tested.
    * ``n_channels : Sequence[int]`` : The number of the channels to be tested
    * ``n_iter : int`` : Number of iteration per error parameter point tested.
    * ``layout : 'CHW'|'HWC'``. Layout of the tensors where the error models are generated. CHW is channel first, HWC is channel last.
    """

    if os.path.isdir(models_path):
        all_models_paths = [
            os.path.join(models_path, file) for file in os.listdir(models_path)
        ]
    else:
        all_models_paths = [models_path]
    total_progr = len(channel_sizes) * len(n_channels) * len(all_models_paths)
    pbar = tqdm(total=total_progr)


    if output_path is not None:
        image_output_folder_path = os.path.join(output_path, 'images')
        os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    else:
        image_output_folder_path = None

    for file_path in all_models_paths:
        filename = os.path.basename(file_path)
        error_model = ErrorModel.from_json_file(file_path)
        f_gen = FaultGenerator(error_model, generator_mapping=GENERATOR_MAPPING)
        for shape in channel_sizes:
            for n_channel in n_channels:
                test_name = f'{filename[:-5]}_{n_channel}_{shape[0]}_{shape[1]}'
                pbar.set_description(
                    f"{filename[:-5]} {n_channel}x{shape[0]}x{shape[1]}"
                )
                if layout == "CHW":
                    out_shape = (n_channel,) + tuple(shape)
                elif layout == "HWC":
                    out_shape = tuple(shape) + (n_channel,)
                test_fault_generator(
                    f_gen, out_shape, image_output_folder_path, test_name, layout, pbar=pbar, n_attempts=n_iter
                )
                pbar.update(1)


def test_fault_generator(
    fault_gen: FaultGenerator,
    output_shape : Sequence[int],
    image_output_folder_path : Optional[str],
    test_name : str,
    layout : Literal['CHW','HWC']="CHW",
    n_attempts=100,
    pbar: Optional[tqdm] = None,
):
    count = 0
    for sp_name, gen_f, sp_params in fault_gen.error_model.spatial_patterns_generator():
        if pbar is not None:
            pbar.set_postfix_str(sp_name)
        flat_params = {**sp_params["keys"], **sp_params["stats"]}
        for tr in range(n_attempts):
            try:
                is_random = any(
                    isinstance(val, dict) and "RANDOM" in val
                    for val in flat_params.values()
                )
                if is_random:
                    realized_params = ErrorModelEntry.realize_random_parameters(
                        flat_params
                    )
                else:
                    realized_params = flat_params
                mask = gen_f(output_shape, realized_params, layout)
                if image_output_folder_path is not None:
                    image_path = os.path.join(image_output_folder_path, f'{test_name}_{sp_name}_{count}_{tr}.png')
                    plot_mask(mask, layout_type=layout, output_path=image_path, save=True, show=False, invalidate=True, description=str(sp_params))         
            except Exception as e:

                print(f"Failed generation")
                print(tb.format_exc())
                print(f"Attempt number: {tr}")
                print(f"Pattern: {sp_name}")
                print(f"Shape: {output_shape}")
                print(f"Layout: {layout}")
                print(f"Original params: {sp_params}")
                print(f"Realized Params: {realized_params}")
                raise e
        count += 1

def parse_args():
    parser = argparse.ArgumentParser(
        "Pattern Generator Tester",
        description="""
            Tool for testing the correctness of the error model pattern generators.

            The tool loads one or more error model in the classes .json format and tries to generate each configuration
            of spatial class and spatial parameters in the error models. The tool can also visualize the generated patterns
            helping the developer to check if the patterns are generated correctly.
        """,
    )

    parser.add_argument(
        "error_model_path",
        type=str,
        help="Path to a folder of .json error models or to a single .json error model file.",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        required=False,
        type=str,
        help="Output path for the test report and images. Specifying this enables visualization",
    )

    parser.add_argument(
        "-l",
        "--layout",
        default="CHW",
        type=str,
        help="Order of the tensors dimensions. For now CHW (PyTorch ordering) and HWC (TensorFlow ordering) are supported",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        default=5,
        type=int,
        help="Number of masks generated for each possible configuration of spatial parameters. Default is 5.",
    )

    parser.add_argument('-d', '--channel-sizes', nargs='+', help='List of channel sizes to evaluate, specified as space separated tuples of two items each representing channel height and width. Example: -d 11,11 22,22 40,40', default=["1,1", "7,7", "16,16", "20,20", "32,32", "416,416", "128,64", "1,100", "100,1"]) 

    parser.add_argument('-c', '--n-channels', type=int, nargs='+', help='List of number of channels, specified as space separated integers. Example: -c 1 2 10 20', default=[1, 3, 4, 12, 32, 100, 128, 137, 256])
    
    args = parser.parse_args()

    new_channel_sizes = []
    for i, ch in enumerate(args.channel_sizes):
        members = ch.split(',')
        if len(members) != 2:
            parser.error(f'Element {ch} at position {i} could not be parsed in --channel-sizes could not be parsed. (Tuples of two comma separated integeres are required)')
        a, b = members
        a, b = int(a), int(b)
        if a <= 0 or b <= 0:
            parser.error(f'An element of {ch} at position {i} is non-positive. All channel in --channel-sizes dimensions must be strictly positive.')
        new_channel_sizes.append((a, b))

    args.channel_sizes = new_channel_sizes
    return args


if __name__ == "__main__":
    args = parse_args()
    test_error_models(args.error_model_path, output_path=args.output_path, n_iter=args.iterations, layout=args.layout, channel_sizes=args.channel_sizes, n_channels=args.n_channels)
    print('Test completed successfully')