from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from classes.error_models.error_model import ErrorModel
from classes.fault_generator.fault import Fault, FaultBatch
from classes.value_generators.value_class_distribution import ValueClassDistribution
from classes.pattern_generators import get_default_generators, PatternGenerator

import numpy as np

from classes.value_generators.value_class import ValueClass


@dataclass
class FaultGenerator:
    error_model: ErrorModel
    generator_mapping: Mapping[str, PatternGenerator] = field(
        default_factory=get_default_generators
    )
    layout: str = "CHW"
    fixed_spatial_class: Optional[str] = None
    fixed_spatial_parameters: Optional[Dict[str, Any]] = None
    fixed_domain_class: Optional[ValueClassDistribution] = None

    def __post_init__(self):
        if (
            self.fixed_spatial_class
            and not self.fixed_spatial_parameters
            and self.fixed_spatial_class not in self.error_model.entries_name
        ):
            raise ValueError(
                f"Entry name {self.fixed_spatial_class} does not exist in error model, and no spatial parameters are specified"
            )

    def generate_mask(
        self,
        output_shape: Sequence[int],
        value_range=np.array([-30.0, 30.0], dtype=np.float32),
        dtype=None,
    ):
        # Data Type is inferred from value range if not specified
        if dtype is None:
            dtype = value_range.dtype

        if self.fixed_spatial_class:
            # Use spatial class fixed at the creation of the generator
            spatial_pattern_name = self.fixed_spatial_class
            random_entry = self.error_model.get_entry_by_name(spatial_pattern_name)
        else:
            # Pick random entry (spatial pattern) using the frequency
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

        # Get the pattern generating function from the dictionary by name
        pattern_generator_fn = self.generator_mapping[spatial_pattern_name]
        # Generate the mask containing corrupted values with the same shape of the output
        # The mask contains 1 when the values are corrupted
        corrupted_value_mask = pattern_generator_fn(
            output_shape, sp_parameters, self.layout
        )
        # Get the number of corrupted values
        corrupted_values_count = int(corrupted_value_mask.sum())
        # Generate a list of value classes ids of lenght equal to the number of corrupted values
        domain_class_mask = domain_class.generate_value_classes(
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
        return Fault(
            corrupted_value_mask, corrupted_values, spatial_pattern_name, sp_parameters
        )

    def generate_batched_mask(
        self,
        output_shape: Sequence[int],
        batch_size: int,
        value_range=np.array([-30.0, 30.0], dtype=np.float32),
        dtype=None,
    ):
        masks = []
        values = []
        sp_classes = []
        sp_parameters = []

        assert batch_size > 0, "batch_size must be a positive int"

        for i in range(batch_size):
            mask, corr_value, sp_class, sp_parameter = self.generate_mask(
                output_shape, value_range, dtype=dtype
            )
            masks.append(mask)
            values.append(corr_value)
            sp_classes.append(sp_class)
            sp_parameters.append(sp_parameter)
        masks = np.stack(masks, axis=0)
        values_lengths = np.array(list(map(np.size, values)), dtype=np.intp)
        values_index = np.zeros(values_lengths.size + 1, dtype=np.intp)
        values_index[1:] = np.cumsum(values_lengths)
        values = np.concatenate(values, axis=0)

        return FaultBatch(masks, values, values_index, sp_classes, sp_parameters)
