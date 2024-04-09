from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from classes.error_models.error_model_entry import ErrorModelEntry
from classes.utils import random_choice_safe


@dataclass
class ErrorModel:
    entries_name: Sequence[str]
    entries: Sequence[ErrorModelEntry]
    entries_counts: Sequence[int]
    file_path: Optional[str] = None

    def __post_init__(self):
        if not (
            len(self.entries_name) == len(self.entries) == len(self.entries_counts)
        ):
            raise ValueError(
                "entries_name, entries and entries_counts must have the same length"
            )

    @staticmethod
    def from_json_dict(
        json_dict: Dict[str, Any], path: Optional[str] = None
    ) -> ErrorModel:
        entries_names = []
        entries = []
        entries_counts = []
        for pattern_name, entry_dict in json_dict.items():
            if not pattern_name.startswith("_"):
                entries_names.append(pattern_name)
                entries.append(
                    ErrorModelEntry.from_json_object(pattern_name, entry_dict)
                )
                entries_counts.append(entry_dict["count"])
        return ErrorModel(entries_names, entries, entries_counts, file_path=path)

    @staticmethod
    def from_json_file(path: str) -> ErrorModel:
        with open(path, "r") as f:
            json_dict = json.load(f)
            return ErrorModel.from_json_dict(json_dict, path=path)

    @staticmethod
    def from_model_folder(folder_path: str, operator: str) -> ErrorModel:
        file_path = os.path.join(folder_path, f"{operator}.json")
        return ErrorModel.from_json_file(file_path)

    def get_entry_by_name(self, name: str) -> ErrorModelEntry:
        idx = self.entries_name.index(name)
        return self.entries[idx]

    def realize_entry(self) -> ErrorModelEntry:
        choice_idx = random_choice_safe(len(self.entries), p=self.entries_counts)
        return self.entries[choice_idx]

    def __repr__(self) -> str:
        entries_repr = []
        for en, e, ec in zip(self.entries_name, self.entries, self.entries_counts):
            entries_repr.append(f"(pattern_name:{en}, entry:{e}, count:{ec})")
        entries = ",".join(entries_repr)
        return f"Error Model(file_path={self.file_path}, [{entries}])"

    def __get_item__(self, index: int) -> Tuple[str, ErrorModelEntry, int]:
        return (
            self.entries_name[index],
            self.entries[index],
            self.entries_counts[index],
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, x):
        if isinstance(x, str):
            return x in self.entries_name
        else:
            raise TypeError("Only str argumetns are supported for __contains__")


    def spatial_patterns_generator(self):
        """
        Creates a Generator that enumerates all the combinations of ErrorModelEntry and spatial parameters
        present in the error model.
        Useful for testing that all the possible parameters contained in the error model work well

        Returns
        ---
        A Generator, that generates tuples. Each tuple contains in order:
        * The name of the error model entry (or spatial pattern)
        * The generator function of that spatial pattern
        * A str-indexed dict that contains the spatial parameters.
        """
        for entry in self.entries:
            entry: ErrorModelEntry
            for sp_parameter in entry.spatial_parameters:
                yield (
                    entry.spatial_pattern_name,
                    self.generator_mapping[entry.spatial_pattern_name],
                    sp_parameter,
                )