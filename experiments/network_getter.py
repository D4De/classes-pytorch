import os
import torch

from functools import partial
from collections import namedtuple

NetworkInfo = namedtuple('NetworkInfo', ['task', 'num_classes', 'csv_header'])

# available network/dataset combinations, along with the task they accomplish and the number of classes (or other info if not classification); the
# final item is the csv header to use for the corrupted tensor report
available = {
    'res50_cifar10':       NetworkInfo('Classification', 10, ['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'Topclass golden', 'Topclass corrupted', 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']),
    'alexnet_cifar10':     NetworkInfo('Classification', 10, ['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'Topclass golden', 'Topclass corrupted', 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']),
    'mobilenetv2_gtsrb':   NetworkInfo('Classification', 43, ['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'Topclass golden', 'Topclass corrupted', 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']),
    'yolov11_coco':        NetworkInfo('Detection',      80, ['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'Precision', 'Recall']),
    'deeplabv3_oxfordpet': NetworkInfo('Segmentation',    3, ['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'mIOU', 'Precision', 'Recall']),
}

def requires_single_metrics(id: str):
    return id == 'yolov11_coco'

def get_network_and_exp_functions(id: str, batch_size: int, device, return_model_only=False):
    """id must be in the form 'networkname_datasetname'"""
    
    # find corresponding network tuple among the available ones
    if id not in available:
        raise ValueError(f'{id} is not an available combination. Choose from {list(available.keys())}.')
    network_info = available[id]

    weights_dir = 'experiments/weights'
    data_dir = 'experiments/dataset_data'

    match id:
        case 'res50_cifar10':
            from nets_repo.classification.cifar10.models.resnet import ResNet50
            from nets_repo.classification.cifar10.dataset import getCIFAR10
            from experiments.network_runner import classification_run
            from experiments.metrics import compute_classification_run_metrics
            model = ResNet50()
            model.load_state_dict(torch.load(os.path.join(weights_dir, 'resnet50', 'fp32_res50_cifar10.pth')))
            if not return_model_only:
                _, loader, _ = getCIFAR10(os.path.join(data_dir, 'cifar10'), (32,32), batch_size, shuffle_test=True)
                run_fn = partial(classification_run, model=model, dataloader=loader, device=device, num_classes=network_info.num_classes)
                metrics_fn = compute_classification_run_metrics

        case 'alexnet_cifar10':
            from nets_repo.classification.cifar10.models.alexnet import AlexNet
            from nets_repo.classification.cifar10.dataset import getCIFAR10
            from experiments.network_runner import classification_run
            from experiments.metrics import compute_classification_run_metrics
            model = AlexNet()
            model.load_state_dict(torch.load(os.path.join(weights_dir, 'alexnet', 'fp32_alexnet_cifar10.pth')))
            if not return_model_only:
                _, loader, _ = getCIFAR10(os.path.join(data_dir, 'cifar10'), (32,32), batch_size, shuffle_test=True)
                run_fn = partial(classification_run, model=model, dataloader=loader, device=device, num_classes=network_info.num_classes)
                metrics_fn = compute_classification_run_metrics

        case 'mobilenetv2_gtsrb':
            from other_nets.classification.gtsrb.models.mobilenetv2 import get_mobilenetv2_model
            from other_nets.classification.gtsrb.dataset import getGTSRB
            from experiments.network_runner import classification_run
            from experiments.metrics import compute_classification_run_metrics
            model = get_mobilenetv2_model(43, os.path.join(weights_dir, 'mobilenetv2', 'mobilenetv2_gtsrb_best.pth'))
            if not return_model_only:
                loader = getGTSRB(os.path.join(data_dir, 'gtsrb'), batch_size, 0)
                run_fn = partial(classification_run, model=model, dataloader=loader, device=device, num_classes=network_info.num_classes)
                metrics_fn = compute_classification_run_metrics

        case 'yolov11_coco':
            from other_nets.detection.coco.models.yolov11.yolov11 import get_yolov11
            from other_nets.detection.coco.dataset import getCOCO
            from experiments.network_runner import yolo_detection_run
            from experiments.metrics import compute_yolo_detection_run_metrics
            image_size: int = 128
            model = get_yolov11(os.path.join(weights_dir, 'yolov11'))
            if not return_model_only:
                loader = getCOCO(os.path.join(data_dir, 'coco'), image_size, batch_size)

                # load cocodetection to ultralytics id mapping for the run function
                import yaml
                with open('other_nets/detection/coco/cocodetection_ids_to_ultralytics_ids.yaml') as f:
                    id_mapping = yaml.load(f, yaml.SafeLoader)

                run_fn = partial(yolo_detection_run, model=model, dataloader=loader, batch_size=batch_size, id_mapping=id_mapping)
                metrics_fn = compute_yolo_detection_run_metrics

        case 'deeplabv3_oxfordpet':
            from other_nets.segmentation.oxfordpet.models.deeplabv3 import get_deeplabv3
            from other_nets.segmentation.oxfordpet.dataset import get_oxfordpet
            from experiments.network_runner import segmentation_run
            from experiments.metrics import compute_segmentation_run_metrics
            model = get_deeplabv3(os.path.join(weights_dir, 'deeplabv3', 'deeplabv3_pet_0.7500.pt'))
            if not return_model_only:
                loader = get_oxfordpet(os.path.join(data_dir, 'oxfordpet'), batch_size, 0)
                run_fn = partial(segmentation_run, model=model, dataloader=loader, device=device)
                metrics_fn = partial(compute_segmentation_run_metrics, num_classes=network_info.num_classes)

        case _:
            raise ValueError(f'Unknown network_dataset id {id}')

    model.eval()
    model.to(device=device)
    if return_model_only:
        return model

    return model, loader, network_info, run_fn, metrics_fn