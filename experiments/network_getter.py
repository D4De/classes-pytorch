import os
import torch

from functools import partial
from collections import namedtuple

NetworkInfo = namedtuple('NetworkInfo', ['task', 'num_classes', 'csv_header'])

# available network/dataset combinations, along with the task they accomplish and the number of classes (or other info if not classification); the
# final item is the csv header to use for the corrupted tensor report
available = {
    'resnet50_cifar10':    NetworkInfo('Classification', 10, ['Layer name', 'Error number', 'Sample index', 'Spatial pattern', 'Topclass golden', 'Topclass corrupted', 'Ranking deviation present', 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']),
    'alexnet_cifar10':     NetworkInfo('Classification', 10, ['Layer name', 'Error number', 'Sample index', 'Spatial pattern', 'Topclass golden', 'Topclass corrupted', 'Ranking deviation present', 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']),
    'mobilenetv2_gtsrb':   NetworkInfo('Classification', 43, ['Layer name', 'Error number', 'Sample index', 'Spatial pattern', 'Topclass golden', 'Topclass corrupted', 'Ranking deviation present', 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']),
    'yolov11_coco':        NetworkInfo('Detection', 80, ['Layer name', 'Error number', 'Sample index', 'Spatial pattern', 'Precision', 'Recall']),
    'deeplabv3_oxfordpet': NetworkInfo('Segmentation', 3, ['Layer name', 'Error number', 'Sample index', 'Spatial pattern', 'mIOU', 'Precision', 'Recall']),
}

def get_network_and_exp_functions(id: str, batch_size: int, device):
    """id must be in the form 'networkname_datasetname'"""
    
    # find corresponding network tuple among the available ones
    if id not in available:
        raise ValueError(f'{id} is not an available combination. Choose from {available}.')
    network_info = available[id]

    weights_dir = 'experiments/weights'
    data_dir = 'experiments/dataset_data'

    match id:
        case 'resnet50_cifar10':
            from experiments.exp_resnet50_gtsrb.model import get_model
            from nets_repo.classification.cifar10.dataset import getCIFAR10
            from experiments.network_runner import classification_golden_run, classification_error_run
            from experiments.metrics import compute_classification_golden_run_metrics, compute_classification_final_metrics
            model = get_model(10, os.path.join(weights_dir, 'resnet50'))
            _, loader, _ = getCIFAR10(os.path.join(data_dir, 'cifar10'), (32,32), batch_size)
            golden_run_fn = partial(classification_golden_run, model=model, dataloader=loader, num_classes=network_info.num_classes, device=device)
            golden_run_metrics_fn = partial(compute_classification_golden_run_metrics, num_classes=network_info.num_classes)
            error_run_fn = partial(classification_error_run, model=model, dataloader=loader, num_classes=network_info.num_classes, device=device)
            error_run_metrics_fn = compute_classification_final_metrics

        case 'alexnet_cifar10':
            from nets_repo.classification.cifar10.models.alexnet import AlexNet
            from nets_repo.classification.cifar10.dataset import getCIFAR10
            from experiments.network_runner import classification_golden_run, classification_error_run
            from experiments.metrics import compute_classification_golden_run_metrics, compute_classification_final_metrics
            model = AlexNet()
            model.load_state_dict(torch.load(os.path.join(weights_dir, 'alexnet/fp32_alexnet_cifar10.pth')))
            _, loader, _ = getCIFAR10(os.path.join(data_dir, 'cifar10'), (32,32), batch_size)
            golden_run_fn = partial(classification_golden_run, model=model, dataloader=loader, num_classes=network_info.num_classes, device=device)
            golden_run_metrics_fn = partial(compute_classification_golden_run_metrics, num_classes=network_info.num_classes)
            error_run_fn = partial(classification_error_run, model=model, dataloader=loader, num_classes=network_info.num_classes, device=device)
            error_run_metrics_fn = compute_classification_final_metrics

        case 'mobilenetv2_gtsrb':
            from other_nets.classification.gtsrb.models.mobilenetv2 import get_mobilenetv2_model
            from other_nets.classification.gtsrb.dataset import getGTSRB
            from experiments.network_runner import classification_golden_run, classification_error_run
            from experiments.metrics import compute_classification_golden_run_metrics, compute_classification_final_metrics
            model = get_mobilenetv2_model(43, os.path.join(weights_dir, 'mobilenetv2', 'mobilenetv2_gtsrb_best.pth'))
            loader = getGTSRB(os.path.join(data_dir, 'gtsrb'), batch_size, 0)
            golden_run_fn = partial(classification_golden_run, model=model, dataloader=loader, num_classes=network_info.num_classes, device=device)
            golden_run_metrics_fn = partial(compute_classification_golden_run_metrics, num_classes=network_info.num_classes)
            error_run_fn = partial(classification_error_run, model=model, dataloader=loader, num_classes=network_info.num_classes, device=device)
            error_run_metrics_fn = compute_classification_final_metrics

        case 'yolov11_coco':
            from other_nets.detection.coco.models.yolov11.yolov11 import get_yolov11
            from other_nets.detection.coco.dataset import getCOCO
            from experiments.network_runner import yolo_detection_golden_run, yolo_detection_error_run
            from experiments.metrics import compute_yolo_detection_golden_run_metrics, compute_yolo_detection_final_metrics
            model = get_yolov11(os.path.join(weights_dir, 'yolov11'))
            loader = getCOCO(os.path.join(data_dir, 'coco'), batch_size)
            golden_run_fn = partial(yolo_detection_golden_run, model=model, dataloader=loader)
            golden_run_metrics_fn = compute_yolo_detection_golden_run_metrics
            error_run_fn = partial(yolo_detection_error_run, model=model, dataloader=loader)
            error_run_metrics_fn = compute_yolo_detection_final_metrics

        case 'deeplabv3_oxfordpet':
            from other_nets.segmentation.oxfordpet.models.deeplabv3 import get_deeplabv3
            from other_nets.segmentation.oxfordpet.dataset import get_oxfordpet
            from experiments.network_runner import segmentation_golden_run, segmentation_error_run
            from experiments.metrics import compute_segmentation_golden_run_metrics, compute_segmentation_final_metrics
            model = get_deeplabv3(os.path.join(weights_dir, 'deeplabv3', 'deeplabv3_pet_0.7500.pt'))
            loader = get_oxfordpet(os.path.join(data_dir, 'oxfordpet'), batch_size, 0)
            golden_run_fn = partial(segmentation_golden_run, model=model, dataloader=loader, device=device)
            golden_run_metrics_fn = partial(compute_segmentation_golden_run_metrics, num_classes=network_info.num_classes)
            error_run_fn = partial(segmentation_error_run, model=model, dataloader=loader, device=device)
            error_run_metrics_fn = partial(compute_segmentation_final_metrics, num_classes=network_info.num_classes)

    model.eval()
    model.to(device=device)
    return model, loader, network_info, golden_run_fn, golden_run_metrics_fn, error_run_fn, error_run_metrics_fn