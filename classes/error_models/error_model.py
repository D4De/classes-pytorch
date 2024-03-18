from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Sequence

from classes.error_models.error_model_entry import ErrorModelEntry
from classes.utils import random_choice_safe


@dataclass
class ErrorModel:
    entries_name: Sequence[str]
    entries: Sequence[ErrorModelEntry]
    entries_counts: Sequence[int]

    def __post_init__(self):
        if not (
            len(self.entries_name) == len(self.entries) == len(self.entries_counts)
        ):
            raise ValueError(
                "entries_name, entries and entries_counts must have the same length"
            )

    @staticmethod
    def from_json_dict(json_dict: Dict[str, Any]) -> ErrorModel:
        entries_names = []
        entries = []
        entries_counts = []
        for pattern_name, entry_dict in json_dict.items():
            if not pattern_name.startswith("_"):
                entries_names.append()
                entries.append(
                    ErrorModelEntry.from_json_object(pattern_name, entry_dict)
                )
                entries_counts.append(entry_dict["count"])
        return ErrorModel(entries, entries_counts)

    @staticmethod
    def from_json_file(path: str) -> ErrorModel:
        with open(path, "r") as f:
            json_dict = json.load(f)
            return ErrorModel.from_json_dict(json_dict)

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
