import torch
import numpy as np
import tarfile
import pathlib

from torch.utils.data import Dataset

from classes.simulators.pytorch.pytorch_fault import PyTorchFault, PyTorchFaultBatch


def fault_collate_fn(data: list[PyTorchFaultBatch]):
    names, fault_ids, masks, values, sp_classes, sp_parameters = zip(*data)

    torch_values_idxs = torch.LongTensor(
        [value_tensor.size() for value_tensor in values]
    )

    batch_masks = torch.stack(masks)
    batch_values = torch.concat(values)

    batch_values_idxs = torch.LongTensor(
        torch.zeros(len(masks) + 1).long()
    )  # sono veramente long
    batch_values_idxs[1:] = torch.cumsum(torch_values_idxs.squeeze(), dim=0)

    return PyTorchFaultBatch(
        names,
        fault_ids,
        batch_masks,
        batch_values,
        batch_values_idxs,
        sp_classes,
        sp_parameters,
    )


class FaultListFromTarFileDynamic(Dataset[PyTorchFault]):
    def __init__(
        self,
        tarf: tarfile.TarFile,
        module_name: str,
        num_faults: int,
        fault_batch_size: int,
    ) -> None:
        super().__init__()
        self.tarf = tarf
        self.module_name = module_name
        self.num_faults = num_faults
        self.fault_batch_size = fault_batch_size

        self.include_sp_parameters = True
        self.fault_list_info = None
        self.tmp_path = None

    def __getitem__(self, index: int) -> PyTorchFault:
        file_idx  = index // self.fault_batch_size
        batch_idx = index % self.fault_batch_size
        
        # fault_file_path = os.path.join(
        #     selected_module_name, f"{file_idx}_faults_{selected_module_name}.npz"
        # )
        
        # tarfile.py only supports posix pathfiles: force the path format to posix, even on Windows
        fault_file_path = pathlib.Path(f"{self.module_name}/{file_idx}_faults_{self.module_name}.npz").as_posix()

        member_file = self.tarf.extractfile(fault_file_path)
        if not member_file:
            raise FileNotFoundError(
                f"File {fault_file_path} not found in fault list archive"
            )
        try:
            fault_list_file = np.load(member_file, allow_pickle=True)
            mask = fault_list_file["masks"][batch_idx]
            slice_begin, slice_end = fault_list_file["values_index"][
                batch_idx : batch_idx + 2
            ]
            values = fault_list_file["values"][slice_begin:slice_end]

            torch_mask = torch.from_numpy(mask)
            torch_values = torch.from_numpy(values)

            sp_class = fault_list_file["sp_classes"][batch_idx]
            sp_param = fault_list_file["sp_parameters"][batch_idx]
            return PyTorchFault(
                self.module_name,
                index,
                torch_mask,
                torch_values,
                sp_class,
                sp_param,
            )
        finally:
            member_file.close()


    def collate_fn(self, data: list[PyTorchFaultBatch]):
        return fault_collate_fn(data)

    def __len__(self) -> int:
        return self.num_faults
