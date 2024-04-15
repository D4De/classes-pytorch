from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple
from classes.utils import random_choice_safe

from classes.value_generators.value_class import ValueClass

import numpy as np


class ValueClassDistribution(ABC):
    def __init__(self, value_classes: Sequence[ValueClass]) -> None:
        self.value_classes = list(value_classes)

    @abstractmethod
    def generate_value_classes(self, output_shape: Sequence[int]) -> np.ndarray:
        """
        Generates randomly a numpy array of shape ``output_shape`` that following the
        distribution. Each subclass of ``ValueClassDistribution`` must implement
        this method.

        Args
        ---
        * ``output_shape: Sequence[int]``: The shape of the output array

        Returns
        ---
        A numpy array of shaoe ``output_shape`` of dtype ``uint8``.
        Each integer of the array represents the numeric id of the dataclass, accordingly
        to the ids specified in the definition of ``ValueClass`` enum.
        """
        pass

    def get_value_classes(self) -> List[ValueClass]:
        """
        Returns the list of all ``ValueClasses`` that the distribution may generate.
        """
        return list(self.value_classes)

    def get_value_classes_ids(self) -> List[int]:
        """
        Returns the list of ``type_id``s that the distribution may generate.
        This are the only numbers that the array returned by the method ``ValueClassDistribution.get_value_classes``
        can contain.
        """
        return [v.type_id for v in self.value_classes]

    @staticmethod
    def from_json_object(json_dict: Dict[str, Any]) -> ValueClassDistribution:
        """
        Create a ``ValueClassDistribution`` object from a dict created from a json dictionary

        The type of distribution depends from the structure of the json.

        The supported types of ``ValueClassDistribution`` in the json are:
        * ``SingleTypeDistribution``
            Generates value classes of only a single type

            JSON example:

            ```
            {
                "<type>": [
                    100.0,
                    100.0
                ],
                "count": 379,
                "frequency": 0.3832153690596562
            },
            ```
        * ``DoubleTypeDistribution``
            Generates values of two types. The ranges follow the probabilities
            specified in the ranges inside the json.

            JSON example:

            ```
            {
                "<type_1>": [75.0,87.5],
                "<type_2>": [12.5,25.0],
                "count": 68,
                "frequency": 0.022696929238985315
            }
            ```
            The lower end of the range <type_1> with the higher end of the range of <type_2> must sum to 100.0.
            The same must be valid for the higher end of range <type_1> with the the lower end of <type_2>
        * ``RandomDistribution``
            Generates values of multiple type without ranges.
            Only a puntual probability is specified for each class

            JSON Example:
            ```
            {
                "random": [100.0, 100.0],
                "values": {
                    "<type_1>": 0.48441274954435637,
                    "<type_2>": 0.471856059000551,
                    "<type_3>": 0.03941847158055355,
                    "<type_4>": 0.004312719874539058
                },
                "count": 186,
                "frequency": 0.062082777036048066
            }
            ```

            The probabilities inside the object with key "values" must sum to 100.0

        In every JSON example, <type_1>,<type_2>,... must be the replaced with one of the members of ``ValueClass`` enum.

        Args
        ----
        * json_dict: Dict[str, Any]. A dict that represents a JSON object
            that represents a ``ValueDistribution`` as specified above

        Returns
        ---
        An object of a subclass ``ValueClassDistribution``, parsed from ``json_dict``.

        """
        all_keys = set(v.display_name for v in ValueClass)
        if "random" in json_dict:
            value_classes = list(
                map(ValueClass.from_display_name, json_dict["values"].keys())
            )
            freqs = list(json_dict["values"].values())
            return RandomDistribution(value_classes, freqs)
        # Filters out the non value_classes keys in the json, such as "count" and "frequency"
        value_classes_names = all_keys & set(json_dict.keys())
        value_classes = list(map(ValueClass.from_display_name, value_classes_names))
        if len(value_classes) == 1:
            value_class = list(value_classes)[0]
            return SingleTypeDistribution(value_class)
        elif len(value_classes) == 2:
            value_classes = list(value_classes)
            ranges = [json_dict[val_class] for val_class in value_classes_names]
            return DoubleTypeDistribution(value_classes, ranges)
        elif len(value_classes) > 2:
            raise ValueError(
                "Non-Random Domain Classes with more than two maule classes are not supported"
            )
        else:
            raise ValueError("No Value Classes specified in Domain Class definition")


class SingleTypeDistribution(ValueClassDistribution):
    """
    A ``ValueClassDistribution`` consisting of a single ``ValueClass``.
    This class always picks the same ``value_class``.
    """

    def __init__(self, value_class: ValueClass) -> None:
        super().__init__([value_class])
        self.value_class = value_class

    def get_value_class(self) -> ValueClass:
        return self.value_class

    def generate_value_classes(self, output_shape: Sequence[int]) -> np.ndarray:
        arr = np.full(output_shape, fill_value=self.value_class.type_id, dtype=np.uint8)
        return arr

    def __repr__(self):
        return f"SingleTypeDistribution({self.value_class!s}=[100%,100%])"


class DoubleTypeDistribution(ValueClassDistribution):
    def __init__(
        self,
        value_classes: Sequence[ValueClass],
        pct_ranges: Sequence[Tuple[float, float]],
    ) -> None:
        super().__init__(value_classes)
        if len(value_classes) > 2:
            raise NotImplementedError(
                "Probability Range Domain classes with more than 2 value classes are not yet supported"
            )
        if len(value_classes) != len(pct_ranges):
            raise ValueError(
                f"value_classes and pct_ranges must be sequences of the same length. Instead found value_classes of len {len(value_classes)} and pct_ranges of len {len(pct_ranges)}"
            )
        self.pct_ranges = pct_ranges

    def generate_value_classes(self, output_shape: Sequence[int]) -> np.ndarray:
        min_class_1, max_class_1 = self.pct_ranges[0]
        pct_class_1 = np.random.uniform(min_class_1, max_class_1)
        pct_class_2 = 100.0 - pct_class_1

        return np.random.choice(
            self.get_value_classes_ids(),
            size=output_shape,
            p=np.array([pct_class_1, pct_class_2]) / 100.0,
        )

    def __repr__(self):
        vc0, vc1 = self.value_classes
        (r0a, r0b), (r1a, r1b) = self.pct_ranges
        return (
            f"DoubleTypeDistribution({vc0!s}=[{r0a}%,{r0b}%],{vc1!s}=[{r1a}%,{r1b}%])"
        )


class RandomDistribution(ValueClassDistribution):
    """
    A ``ValueClassDistribution`` that generates only a single value
    class.
    """

    def __init__(
        self, value_classes: Sequence[ValueClass], freq: Sequence[float]
    ) -> None:
        super().__init__(value_classes)
        if len(value_classes) != len(freq):
            raise ValueError(
                f"value_classes and freq must be sequences of the same length. Instead found value_classes of len {len(value_classes)} and pct_ranges of len {len(freq)}"
            )
        self.freq = freq

    def generate_value_classes(self, output_shape: Sequence[int]) -> np.ndarray:
        return random_choice_safe(
            self.get_value_classes_ids(), output_shape, p=self.freq
        )

    def __repr__(self) -> str:
        random_values = []
        for vc, fr in zip(self.value_classes, self.freq):
            random_values.append(f"{vc!s}={fr:.3f}")
        ",".join(random_values)
        return f"RandomDistribution({random_values})"
