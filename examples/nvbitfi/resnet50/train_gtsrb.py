import torch
import torchvision
import numpy as np

from torch.utils.data import DataLoader

from train_resnet50 import train_resnet50
from augmentation import TransformableSubset
from gtsrb_transforms import data_transform, train_transform

from pathlib import Path


def main():
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    
    dataset_path = Path('./data')
    dataset_path.mkdir(exist_ok=True)
    assert dataset_path.exists(), "data directory does not exist and could not be created"
    
    checkpoints_dir = Path('./checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    assert checkpoints_dir.exists(), "checkpoints directory does not exist and could not be created"
    
    best_epoch_checkpoint_path = checkpoints_dir / Path('resnet50_gtsrb_best.pth')
    latest_epoch_checkpoint_path = checkpoints_dir / Path('resnet50_gtsrb_latest.pth')
    
    batch_size = 256
    num_epochs = 20
    
    print('Getting GTSRB training dataset')
    data = torchvision.datasets.GTSRB(
        dataset_path, split="train", download=True
    )
    
    n_classes = 43

    train_size = int(np.round(0.7 * len(data)).item())
    valid_size = int(np.round(0.3 * len(data)).item())

    gen = torch.Generator().manual_seed(42)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        data, [train_size, valid_size], generator=gen
    )


    train_dataset = TransformableSubset(train_dataset, data_transform=train_transform())
    valid_dataset = TransformableSubset(valid_dataset, data_transform=data_transform())
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True
    )

    print(f'Starting training for {num_epochs} epochs with batch size {batch_size}')
    train_resnet50(
        train_loader,
        valid_loader,
        n_classes,
        epochs=num_epochs,
        restart_checkpoint_path=best_epoch_checkpoint_path,
        learning_rate=2e-3,
        lr_exp_decay=0.95,
        weight_decay=1e-4,
        device=device,
        weight_best_path=best_epoch_checkpoint_path,
        weight_latest_path=latest_epoch_checkpoint_path,
    )


if __name__ == '__main__':
    main()