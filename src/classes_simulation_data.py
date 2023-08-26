from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class ClassesSimulationData:
    layer_name : str
    torch_layer_type : str
    classes_layer_type : str
    input_shape : tuple
    output_shape : tuple
    spatial_pattern : str
    spatial_parameters : dict
    corrupted_values_count : int
    corrupted_channels_count : List[int]
    domain_pattern : dict
    value_range : Tuple[float, float]

    def as_insert_param_dict(self) -> Dict[str, Any]:

        return {
            "layer_name": self.layer_name,
            "torch_layer_type": self.torch_layer_type,
            "classes_layer_type": self.classes_layer_type,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "spatial_pattern": self.spatial_pattern,
            "spatial_parameters": self.spatial_parameters,
            "corrupted_values": self.corrupted_values_count,
            "corrupted_channels_count": self.corrupted_channels_count,
            "domain_pattern": self.domain_pattern,
            "value_range_min": self.value_range[0],
            "value_range_max": self.value_range[1]
        }