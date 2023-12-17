from dataclasses import dataclass
from typing import Callable

from .classes_simulation_data import ClassesSimulationData
from .injection_sites_generator import *
import torch.nn as nn
import torch
import sqlite3

from .loggers import get_logger

logger = get_logger("ErrorSimulator")


class Simulator(nn.Module):
    def __init__(
        self,
        layer_type,
        size=None,
        models_folder="./models",
        fixed_spatial_class=None,
        fixed_domain_class=None,
        enable=True,
    ):
        super().__init__()
        self.sites_count = 1
        self.layer_type = layer_type
        self.size = size
        self.models_mode = ""
        self.models_folder = models_folder
        self.fixed_spatial_class = fixed_spatial_class
        self.fixed_domain_class = fixed_domain_class
        self.enable = enable

    def __generate_injection_sites(
        self, sites_count, layer_type, size, models_folder
    ) -> List[InjectionSite]:
        injection_site = InjectableSite(layer_type, size)

        injection_sites = InjectionSitesGenerator(
            [injection_site],
            models_folder,
            fixed_spatial_class=self.fixed_spatial_class,
            fixed_domain_class=self.fixed_domain_class,
        ).generate_random_injection_sites(sites_count)

        return injection_sites

    def forward(self, x):
        if not self.enable:
            return x
        injection_site = self.__generate_injection_sites(
            1, self.layer_type, self.size or str(tuple(x.shape)), self.models_folder
        )
        range_min = torch.min(x)
        range_max = torch.max(x)
        if len(injection_site) > 0 and len(injection_site[0]) > 0:
            for idx, value in injection_site[0].get_indexes_values():
                x[idx] = float(value.get_value(range_min, range_max))
        else:
            raise RuntimeError("No injection happened")
        return x


def create_capture_func_to_list(dest_list: List[ClassesSimulationData]):
    def _capture_to_list(data: ClassesSimulationData):
        dest_list.append(data)

    return _capture_to_list


def create_simulator_hook(
    torch_type_to_layer_model_map: Dict[str, str],
    layer_name: str,
    fixed_layer_type: Optional[str] = None,
    fixed_size: Optional[Iterable] = None,
    models_folder: str = "./models",
    fixed_spatial_class: Optional[str] = None,
    fixed_domain_class: Optional[str] = None,
    data_capture_func: Optional[Callable[[ClassesSimulationData], None]] = None,
    verbose: bool = False,
):
    def _hook(module, input, output):
        range_min = torch.min(output)
        range_max = torch.max(output)
        input_shape = tuple(input[0].shape)
        output_shape = fixed_size or tuple(output.shape)
        layer_type = (
            fixed_layer_type or torch_type_to_layer_model_map[type(module).__name__]
        )
        injection_site = InjectableSite(layer_type, str(output_shape))

        if verbose:
            print("--- Injection details ---")

        injection_sites, (
            spatial_pattern,
            spatial_parameters,
            domain_pattern,
            corrupted_channel_count,
        ) = InjectionSitesGenerator(
            [injection_site],
            models_folder,
            fixed_spatial_class=fixed_spatial_class,
            fixed_domain_class=fixed_domain_class,
        ).generate_random_injection_sites(
            1, return_pattern_details=True
        )

        if data_capture_func is not None:
            data_capture_func(
                ClassesSimulationData(
                    layer_name,
                    torch_layer_type=type(module).__name__,
                    classes_layer_type=layer_type,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    spatial_pattern=spatial_pattern,
                    spatial_parameters=spatial_parameters,
                    corrupted_values_count=len(injection_sites[0]),
                    corrupted_channels_count=corrupted_channel_count,
                    domain_pattern=domain_pattern,
                    value_range=(range_min.item(), range_max.item()),
                )
            )

        if verbose:
            print(f"Layer Name: {layer_name}")
            print(f"Torch Layer type: {type(module).__name__}")
            print(f"CLASSES Layer Type: {layer_type}")
            print(f"Output Shape: {output_shape}")
            print(f"Spatial Pattern: {spatial_pattern}")
            print(f"Spatial Parameters: {spatial_parameters}")
            print(f"Corrupted Values: {len(injection_sites[0])}")
            print(f"Domain Pattern: {domain_pattern}")
            print(f"In range values: ({range_min}, {range_max})")

        if len(injection_sites) > 0 and len(injection_sites[0]) > 0:
            x = torch.tensor(injection_sites[0].get_indexes())
            y = torch.tensor(injection_sites[0].get_values(range_min, range_max))
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            output[x[:, 0], x[:, 1], x[:, 2], x[:, 3]] = y
            # for idx, value in injection_sites[0].get_indexes_values():
            #    # before = float(output[idx])
            #    output[idx] = float(value.get_value(range_min, range_max))
            # if verbose:
            # print(f'Index {idx} - Domain Type: {value} - Value: {before} -> {output[idx]}')
        else:
            raise RuntimeError("No injection happened")

        return output

    return _hook
