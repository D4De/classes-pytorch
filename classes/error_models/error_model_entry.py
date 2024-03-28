from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
from classes.utils import random_choice_safe

from classes.value_generators.value_class_distribution import ValueClassDistribution


@dataclass
class ErrorModelEntry:
    spatial_pattern_name: str
    domain_classes: Sequence[ValueClassDistribution]
    domain_classes_counts: Sequence[int]
    spatial_parameters: Sequence[Dict[str, Any]]
    spatial_classes_counts: Sequence[int]

    @staticmethod
    def from_json_object(name: str, json_dict: Dict[str, Any]) -> ErrorModelEntry:
        domain_classes_dict = json_dict["domain_classes"]
        sp_parameters_dict = json_dict["parameters"]

        domain_classes_counts = [dom_cl["count"] for dom_cl in domain_classes_dict]
        domain_classes = [
            ValueClassDistribution.from_json_object(dom_cl) for dom_cl in domain_classes_dict
        ]
        spatial_parameters_counts = [
            sp_param["count"] for sp_param in sp_parameters_dict
        ]

        return ErrorModelEntry(
            name,
            domain_classes,
            domain_classes_counts,
            sp_parameters_dict,
            spatial_parameters_counts,
        )
    

    def realize_spatial_parameters(self) -> Dict[str, Any]:
        print(self.spatial_parameters)
        idx_choice = random_choice_safe(
            len(self.spatial_parameters), p=self.spatial_classes_counts
        )

        params = self.spatial_parameters[idx_choice]
        # Condense keys and stats subdivisions in asingle dict
        params = {**params['keys'], **params['stats']}
        is_random = any(
            isinstance(val, dict) and "RANDOM" in val for val in params.values()
        )
        if is_random:
            return self.realize_random_parameters(params)
        else:
            return params
    

    def realize_domain_class(self) -> ValueClassDistribution:
        idx_choice = random_choice_safe(
            len(self.domain_classes), p=self.domain_classes_counts
        )
        return self.domain_classes[idx_choice]

    @staticmethod
    def realize_random_parameters(parameters):
        realized_params: Dict[str, Any] = {}
        # There could be some keys that specify a minumum and a maximum constraint for a parameters (example min_span_width, max_span_width)
        # In this case the values that we randomly extract from a minium or a maximum value must be coherent (the max should be >= than the min)
        # In the models these minimum and maximum constraint can be detected by finding two keys that start min_<X> max_<X> where <X> is a string
        # in common
        min_keys = set(k[4:] for k in parameters.keys() if k.startswith("min_"))
        max_keys = set(k[4:] for k in parameters.keys() if k.startswith("max_"))
        min_max_constrained_parameters = min_keys & max_keys

        # First we insert all the non random keys
        for param_name, param_values in parameters.items():
            if isinstance(param_values, dict) and "RANDOM" in param_values:
                continue
            realized_params[param_name] = param_values
        # then we extract the random ones
        for param_name, param_values in parameters.items():
            if isinstance(param_values, dict) and "RANDOM" in param_values:
                # The param contains random values to be extracted
                random_values = param_values["RANDOM"]
                base_param_name = param_name[4:]
                # Check if the random parameter is constrained by a minimum and a maximum
                if base_param_name in min_max_constrained_parameters:
                    if param_name.startswith("min_"):
                        dual_parameter = f"max_{base_param_name}"
                        constraint = "max"
                    else:
                        dual_parameter = f"min_{base_param_name}"
                        constraint = "min"
                    # If the dual parameters were already generated or inserted in the realized_params dict we have to follow the constaint
                    if dual_parameter in realized_params:
                        dual_parameter_value = realized_params[dual_parameter]
                        # Filter out the random values that not follow the constraints
                        if constraint == "min":
                            eligible_values = [
                                rv for rv in random_values if rv >= dual_parameter_value
                            ]
                        else:
                            eligible_values = [
                                rv for rv in random_values if rv <= dual_parameter_value
                            ]
                        # For avoiding errors we add the constraint if no values were eligible
                        if len(eligible_values) == 0:
                            eligible_values = [dual_parameter_value]
                    else:
                        # We don't have to worry about the constrain, because the dual parameter is not yet realized
                        eligible_values = random_values
                else:
                    eligible_values = random_values
                # eligible_values contains the values that respect the constraints if any
                chosen_index = random_choice_safe(len(eligible_values))
                realized_params[param_name] = eligible_values[chosen_index]

        return realized_params
