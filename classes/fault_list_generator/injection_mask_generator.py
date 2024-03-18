from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from tqdm import tqdm
from classes.error_models.error_model import ErrorModel
from classes.value_generators.domain_class import DomainClass
from classes.pattern_generators import DEFAULT_GENERATORS, PatternGenerator

import numpy as np

from classes.value_generators.value_class import ValueClass


@dataclass
class InjectionMaskGenerator:
    error_model: ErrorModel
    generator_mapping: Dict[str, PatternGenerator] = DEFAULT_GENERATORS
    layout: str = "CHW"
    fixed_spatial_class: Optional[str] = None
    fixed_spatial_parameters: Optional[Dict[str, Any]] = None
    fixed_domain_class: Optional[DomainClass] = (None,)

    def __post_init__(self):
        if (
            self.fixed_spatial_class
            and self.fixed_spatial_class not in self.error_model.entries_name
        ):
            raise ValueError(
                f"Entry name {self.fixed_spatial_class} does not exist in error model"
            )

    def generate_mask(
        self,
        output_shape: Sequence[int],
        value_range=np.array([-30.0, 30.0], dtype=np.float32),
        dtype=None,
    ):
        if dtype is None:
            dtype = value_range.dtype

        if self.fixed_spatial_class:
            spatial_pattern_name = self.fixed_spatial_class
            random_entry = self.error_model.get_entry_by_name(spatial_pattern_name)
        else:
            random_entry = self.error_model.realize_entry()
            spatial_pattern_name = random_entry.spatial_pattern_name

        if self.fixed_domain_class:
            domain_class = self.fixed_domain_class
        else:
            domain_class = random_entry.realize_domain_class()

        if self.fixed_spatial_parameters:
            sp_parameters = self.fixed_spatial_parameters
        else:
            sp_parameters = random_entry.realize_spatial_parameters()

        pattern_generator_f = self.generator_mapping[spatial_pattern_name]
        corrupted_value_mask = pattern_generator_f(
            output_shape, sp_parameters, self.layout
        )

        corrupted_values_count = corrupted_value_mask.sum()
        domain_class_mask = domain_class.generate_domain_classes(
            (corrupted_values_count,)
        )
        corrupted_values = np.zeros_like(domain_class_mask, dtype=dtype)

        for value_class in ValueClass:
            value_class_positions = domain_class_mask == value_class.type_id
            values_to_generate = value_class_positions.sum()
            if values_to_generate > 0:
                corrupted_values[value_class_positions] = value_class.generate_values(
                    (values_to_generate,), value_range, dtype
                )

        corrupted_value_mask[corrupted_value_mask != 0] = domain_class_mask
        return corrupted_value_mask, corrupted_values

    def generate_batched_mask(
        self,
        output_shape: Sequence[int],
        batch_size: int,
        value_range=np.array([-30.0, 30.0], dtype=np.float32),
        dtype=None,
    ):
        masks = []
        values = []
        for i in range(batch_size):
            mask, corr_value = self.generate_mask(
                output_shape, value_range, dtype=dtype
            )
            masks.append(mask)
            values.append(corr_value)
        masks = np.stack(masks, axis=0)
        values = np.concatenate(values, axis=0)
        return masks, values

    def generate_fault_list(
        self,
        n_faults: int,
        output_shape: Sequence[int],
        batch_size: Optional[int] = None,
        value_range=np.array([-30.0, 30.0], dtype=np.float32),
        dtype=None,
        show_progress=True,
    ):
        masks = []
        values = []
        iterations = range(n_faults)
        if show_progress:
            iterations = tqdm(iterations)
        for i in iterations:
            if batch_size:
                masks, values = self.generate_batched_mask(
                    output_shape, batch_size, value_range, dtype=dtype
                )
            else:
                masks, values = self.generate_mask(
                    output_shape, value_range, dtype=dtype
                )
        return masks, values
