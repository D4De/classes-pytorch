from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple
from classes.utils import random_choice_safe

from classes.value_generators.value_class import ValueClass

import numpy as np

class DomainClass(ABC):
    def __init__(self, value_classes : Sequence[ValueClass]) -> None:
        self.value_classes = list(value_classes)
    
    @abstractmethod
    def generate_domain_classes(self, input_shape : Sequence[int]) -> np.ndarray:
        pass

    def get_value_classes(self) -> List[ValueClass]:
        return list(self.value_classes)
    
    def get_value_classes_ids(self) -> List[int]:
        return [v.type_id for v in self.value_classes]
    
    @staticmethod
    def from_json_object(json_dict : Dict[str, Any]) -> DomainClass:
        all_keys = set(v.display_name for v in ValueClass)
        if "random" in json_dict:
            value_classes = list(json_dict["values"].keys())
            freqs = list(json_dict["values"].values())
            return RandomDomainClass(value_classes, freqs)
        value_classes = all_keys & set(json_dict.keys())
        if len(value_classes) == 1:
            value_class = list(value_classes)[0]
            return SingleTypeDomainClass(value_class)
        elif len(value_classes) == 2:
            value_classes = list(value_classes)
            ranges = [json_dict[val_class] for val_class in value_classes]
            return DoubleTypeDomainClass(value_classes, ranges)
        elif len(value_classes) > 2:
            raise ValueError("Non-Random Domain Classes with more than two maule classes are not supported")
        elif len(value_classes) == 0:
            raise ValueError("No Value Classes specified in Domain Class definition")





class SingleTypeDomainClass(DomainClass):
    def __init__(self, value_class : ValueClass) -> None:
        super().__init__([value_class])
        self.value_class = value_class

    def get_value_class(self) -> ValueClass:
        return self.value_class
    
    def generate_domain_classes(self, input_shape : Sequence[int]) -> np.ndarray:
        arr = np.empty(input_shape, dtype=np.uint8)
        arr.fill(self.value_class.type_id)
        return arr

class DoubleTypeDomainClass(DomainClass):
    def __init__(self, value_classes : Sequence[ValueClass], pct_ranges : Sequence[Tuple[float, float]]) -> None:
        super().__init__(value_classes)
        if len(value_classes) > 2:
            raise NotImplementedError('Probability Range Domain classes with more than 2 value classes are not yet supported')
        if len(value_classes) != len(pct_ranges):
            raise ValueError(f'value_classes and pct_ranges must be sequences of the same length. Instead found value_classes of len {len(value_classes)} and pct_ranges of len {len(pct_ranges)}')
        self.pct_ranges = pct_ranges



    def generate_domain_classes(self, input_shape : Sequence[int]) -> np.ndarray:
        min_class_1, max_class_1 = self.pct_ranges[0]
        pct_class_1 = np.random.uniform(min_class_1, max_class_1)
        pct_class_2 = 100.0 - pct_class_1

        return np.random.choice(self.get_value_classes_ids(), 
                                size=input_shape, 
                                p=np.array(pct_class_1, pct_class_2) / 100.0
            )
    

class RandomDomainClass(DomainClass):
    def __init__(self, value_classes : Sequence[ValueClass], freq : Sequence[float]) -> None:
        super().__init__(value_classes)
        if len(value_classes) != len(freq):
            raise ValueError(f'value_classes and pct_ranges must be sequences of the same length. Instead found value_classes of len {len(value_classes)} and pct_ranges of len {len(pct_ranges)}')
        self.freq = freq


    def generate_domain_classes(self, input_shape : Sequence[int]) -> np.ndarray:
        return random_choice_safe(self.get_value_classes_ids(), input_shape, p=self.freq)
    