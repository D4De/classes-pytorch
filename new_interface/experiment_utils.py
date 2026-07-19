import os
import sys
import datetime
from time import time
import torch

#--DIRECTORY UTILS--
def ensure_dir_exists_nonempty(dir: str, config_field: str = None):
    """
    Checks if the provided directory exists and is not empty. Raises an exception if either condition is false.
    If the directory path was read from a configuration file, parameter 'config_field' can be used to point the
    user to which field in the file caused the error.
    """
    check_field_msg = '' if config_field is None else f' Check field {config_field} in configuration file.'

    if not os.path.isdir(dir):
        raise ValueError(f'{dir} is not a directory.{check_field_msg}')
    if len(os.listdir(dir)) == 0:
        raise FileNotFoundError(f'{dir} is empty.{check_field_msg}')
    

def add_dir_to_path(dir: str, config_field: str = None):
    """
    Adds a directory to sys.path. The directory must exist and be nonempty.
    """
    ensure_dir_exists_nonempty(dir, config_field)
    sys.path.insert(0, dir)


#--TIMING UTILS--
def get_stringified_datetime():
    return datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')


class Timer:
    def __init__(self):
        self._start_t: float = 0.0
        self._duration: float = 0.0
    
    def start(self):
        self._start_t = time()

    def stop(self):
        self._duration = time() - self._start_t

    def get_duration_in_seconds(self):
        return self._duration

    def get_duration_in_milliseconds(self):
        return round(self._duration * 1000.0)
    
    def get_duration_as_str(self):
        return str(datetime.timedelta(seconds=self._duration))


#--MISCELLANEOUS UTILS--
def get_network_modules(model: torch.nn.Module):
    modules = {}

    # filter out modules with children
    def _no_children(mod: torch.nn.Module):
        children = mod.named_children()
        try:
            next(children)
        except StopIteration:
            return True
        return False

    for modname, mod in model.named_modules():
        if _no_children(mod):
            modules[modname] = mod
    
    return modules