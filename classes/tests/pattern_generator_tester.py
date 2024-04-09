import os
import traceback as tb

import argparse
from typing import Literal, Optional, Sequence

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
    shapes=[
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
    n_channels=[1, 3, 4, 12, 32, 100, 128, 137, 256],
    n_iter=5,
    layout : Literal['CHW','HWC'] ="CHW",
):

    if os.path.isdir(models_path):
        all_models_paths = [
            os.path.join(models_path, file) for file in os.listdir(models_path)
        ]
    else:
        all_models_paths = [models_path]
    total_progr = len(shapes) * len(n_channels) * len(all_models_paths)
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
        for shape in shapes:
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
                    image_path = os.path.join(image_output_folder_path, f'{test_name}_{sp_name}_{tr}.png')
                    plot_mask(mask, layout_type=layout, output_path=image_path, save=True, show=False, invalidate=True, suptitile=str(sp_params))         
            except Exception as e:

                print(f"Failed generation")
                print(tb.format_exc())
                print(f"Attempt number: {tr}")
                print(f"Pattern: {sp_name}")
                print(f"Shape: {output_shape}")
                print(f"Layout: {layout}")
                print(f"Original params: {sp_params}")
                print(f"Realized Params: {realized_params}")
                exit(1)


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
        help="Output path for the test report and images",
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

    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Generates visualization for each generated mask, saving them to image files.",
    )

    args = parser.parse_args()


    if args.visualize and args.output_path is None:
        parser.error("--visualize requires --output-path to be specified.")

    return args


if __name__ == "__main__":
    args = parse_args()
    test_error_models(args.error_model_path, output_path=args.output_path, n_iter=args.iterations, layout=args.layout)
