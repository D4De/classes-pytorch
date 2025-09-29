import torch
import torchvision

from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm

from model import get_resnet50_model
from augmentation import TransformableSubset
from metrics import accuracy
from gtsrb_transforms import data_transform

from pathlib import Path

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"


def eval_resnet50_gtsrb(weights_path = Path('./checkpoints/resnet50_gtsrb_best.pth')):
    dataset_path = Path('./data')
    batch_size = 512
    
    print('Getting GTSRB dataset')
    data = torchvision.datasets.GTSRB(
        dataset_path, split="train", download=True
    )

    n_classes = 43

    # get an empty ResNet50
    model = get_resnet50_model(
        n_classes, pretrained_weights=False, return_transforms=False
    )

    # load model weights
    print('Loading saved model weights')
    weights = torch.load(weights_path)['model']
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    train_size = int(np.round(0.7 * len(data)).item())
    valid_size = int(np.round(0.3 * len(data)).item())

    gen = torch.Generator().manual_seed(42)

    train_dataset, valid_dataset = torch.utils.data.random_split(
        data, [train_size, valid_size], generator=gen
    )

    valid_dataset = TransformableSubset(
        valid_dataset, data_transform=data_transform()
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True
    )

    valid_count = len(valid_loader.dataset)
    valid_pbar = tqdm(valid_loader)

    print('Starting validation')
    with torch.no_grad():
        scores = torch.zeros((valid_count, n_classes), device=device)
        labels = torch.zeros((valid_count), device=device, dtype=torch.long)

        for batch_id, (image, label) in enumerate(valid_pbar):
            image = image.to(device)
            label = label.to(device)
            predictions: torch.Tensor = model(image)
            start_idx = batch_id * valid_loader.batch_size
            end_idx = start_idx + predictions.size(0)
            scores[start_idx:end_idx, :] = predictions
            labels[start_idx:end_idx] = label

        top_1 = accuracy(scores, labels, k=1)
        top_5 = accuracy(scores, labels, k=5)

    print(f'Top 1: {top_1:.2f}%')
    print(f'Top 5: {top_5:.2f}%')


if __name__ == '__main__':
    eval_resnet50_gtsrb()