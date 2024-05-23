import os
from torch.utils.data import DataLoader

from typing import Literal

from tqdm import tqdm
import numpy as np

from classes.pattern_generators import get_default_generators
from classes.simulators.pytorch.fault_list import PyTorchFaultListInfo
from classes.simulators.pytorch.fault_list_datasets import FaultListFromTarFile
from classes.simulators.pytorch.pytorch_fault import PyTorchFault
from classes.value_generators.value_class import ValueClass
from classes.visualization.mask import plot_mask

# You can change here the mapping with a custom one
GENERATOR_MAPPING = get_default_generators()


def test_fault_list_pytorch(
    fault_list_path: str,
    output_path: str,
    layout: Literal["CHW", "HWC"] = "CHW", 
):
    
    fault_list_info = PyTorchFaultListInfo.load_fault_list_info(fault_list_path)

    image_output_folder_path = os.path.join(output_path, "images")
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

    for module_name in fault_list_info.injectable_layers:
        fault_list_dataset = FaultListFromTarFile(fault_list_path, module_name)
        fault_dataloader = DataLoader(
            fault_list_dataset,
            batch_size=None,
            num_workers=2,
            #collate_fn=fault_list_dataset.collate_fn,
            pin_memory=True,
        )
        for i, fault_batch in enumerate(tqdm(fault_dataloader)):
            fault_batch : PyTorchFault = fault_batch

            mask = fault_batch.corrupted_value_mask
            c, h, w = mask.shape

            image_path = os.path.join(
                image_output_folder_path,
                f"{module_name}_seq_{i}_{c}_{h}_{w}_{fault_batch.spatial_pattern_name}.png",
            )
            description = f'{fault_batch.spatial_pattern_name}\nCount:{mask.sum()}\n{fault_batch.sp_parameters}'
            plot_mask(
                mask.numpy(),
                layout_type=layout,
                output_path=image_path,
                save=True,
                show=False,
                invalidate=True,
                description=description,
                labels=['NO_ERROR'] + [v.display_name for v in ValueClass],
                colors=['white','yellow','orange','red','green','blue']
            )