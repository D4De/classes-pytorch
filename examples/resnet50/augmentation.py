import torch
import torch.functional as FT

import random


class TransformableSubset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, data_transform=None, target_transform=None, fused_transform=None
    ):
        self.dataset = dataset
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.fused_transform = fused_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.data_transform:
            image = self.data_transform(image)
        if self.target_transform:
            label = self.target_transform(image)
        if self.fused_transform:
            image, label = self.fused_transform(image, label)
        return image, label


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [
        FT.adjust_brightness,
        FT.adjust_contrast,
        FT.adjust_saturation,
        FT.adjust_hue,
    ]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ == "adjust_hue":
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255.0, 18 / 255.0)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image
