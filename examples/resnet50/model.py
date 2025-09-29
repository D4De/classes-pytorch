import torch.nn as nn
from torch import hub
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

def get_resnet50_model(n_classes, pretrained_weights=True, return_transforms=True):
    # set torch hub path for downloading weights
    hub_path = Path('E:/University/MasterThesis/pytorch/.cache/torch')
    hub.set_dir(hub_path)
    
    weights = ResNet50_Weights.DEFAULT
    
    if pretrained_weights:
        model = resnet50(weights=weights)
    else:
        model = resnet50()

    # add classification head
    model.fc= nn.Linear(2048, n_classes, bias=True)
    for param in model.fc.parameters():
        param.requires_grad = True

    model.eval()

    if return_transforms:
        return model, weights.transforms()
    else:
        return model