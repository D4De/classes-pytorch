from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Sequence

from classes.error_models.error_model_entry import ErrorModelEntry
from classes.utils import random_choice_safe

@dataclass
class ErrorModel:
    entries : Sequence[ErrorModelEntry]
    entries_counts : Sequence[int]     

    @staticmethod
    def from_json_dict(json_dict : Dict[str, Any]) -> ErrorModel:
        entries = []
        entries_counts = []
        for k, v in json_dict.items():
            if not k.startswith('_'):
                entries.append(ErrorModelEntry.from_json_object(v))
                entries_counts.append(v['count'])
        return ErrorModel(entries, entries_counts)
    

    @staticmethod
    def from_json_file(path : str) -> ErrorModel:
        with open(path, 'r') as f:
            json_dict = json.load(f)
            return ErrorModel.from_json_dict(json_dict)
    
    @staticmethod
    def from_model_folder(folder_path : str, operator : str) -> ErrorModel:
        file_path = os.path.join(folder_path, f'{operator}.json')
        return ErrorModel.from_json_file(file_path)
    
    def realize_entry(self) -> ErrorModelEntry:
        choice_idx = random_choice_safe(len(self.entries), p=self.entries_counts)
        return self.entries[choice_idx]