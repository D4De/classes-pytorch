import torchvision

def data_transform(size=(128,128)):
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size)

        ]
    )
    return data_transforms


def train_transform(size=(128,128)):
    train_transforms = torchvision.transforms.Compose(
        [
            #weights.transforms(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size),
            # torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # torchvision.transforms.RandomGrayscale(),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
    )
    return train_transforms