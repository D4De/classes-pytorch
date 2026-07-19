import torch
import torchvision

from torch.utils.data import DataLoader

from train_resnet50 import train_resnet50

from pathlib import Path


def main():
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    
    dataset_path = Path('./data')
    dataset_path.mkdir(exist_ok=True)
    assert dataset_path.exists(), "data directory does not exist and could not be created"
    
    checkpoints_dir = Path('./checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    assert checkpoints_dir.exists(), "checkpoints directory does not exist and could not be created"
    
    best_epoch_checkpoint_path = checkpoints_dir / Path('resnet50_cifar100_best.pth')
    latest_epoch_checkpoint_path = checkpoints_dir / Path('resnet50_cifar100_latest.pth')
    
    batch_size = 64
    num_epochs = 200
    
    print('Getting CIFAR100 training dataset')
    size = 32
    num_classes = 100

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.Resize(size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f'Starting training for {num_epochs} epochs with batch size {batch_size}')
    train_resnet50(
        trainloader,
        testloader,
        num_classes,
        epochs=num_epochs,
        restart_checkpoint_path=None,
        learning_rate=2e-3,
        lr_exp_decay=0.95,
        weight_decay=1e-4,
        device=device,
        weight_best_path=best_epoch_checkpoint_path,
        weight_latest_path=latest_epoch_checkpoint_path,
    )


if __name__ == '__main__':
    main()