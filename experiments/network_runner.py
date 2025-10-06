import yaml
import torch

from tqdm import tqdm
from ultralytics import YOLO

from classes.simulators.pytorch.simulator_hook import applied_hook


#--GOLDEN RUNS--
def classification_golden_run(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device,
        num_classes: int,
):
    num_samples = len(dataloader) * dataloader.batch_size

    # prepare a golden score matrix: one row per sample, one column per class score
    golden_scores = torch.zeros((num_samples, num_classes), device=device)
    # prepare a golden label matrix: one row per sample, one column for the golden label
    golden_labels = torch.zeros((num_samples), device=device)

    with torch.no_grad():
        for batch_id, (image, label) in enumerate(tqdm(dataloader, desc='Performing golden run', colour='yellow')):
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                # IMPORTANT: image is a batch of samples, so output can be considered a matrix with one row per sample in the batch
                # and 'num_classes' columns listing the probabilities for that sample to correspond to each class

                # save scores and labels
                start_idx = batch_id * dataloader.batch_size
                end_idx = start_idx + output.size(0)
                golden_scores[start_idx:end_idx, :] = output
                golden_labels[start_idx:end_idx] = label
        
    return golden_scores, golden_labels


def segmentation_golden_run(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device,
):
    num_samples = len(dataloader) * dataloader.batch_size
    for sample_img, _ in dataloader: break
    input_shape = sample_img.shape

    # prepare a golden score matrix: one tensor for each sample in the dataloader, each tensor has the same shape as the sample
    golden_scores = torch.zeros((num_samples, input_shape[-2], input_shape[-1]), device=device)
    # prepare a golden label matrix: one row per sample, one column for the golden label
    golden_labels = torch.zeros_like(golden_scores, device=device)

    with torch.no_grad():
        for batch_id, (image, label) in enumerate(tqdm(dataloader, desc='Performing golden run', colour='yellow')):
                image = image.to(device)
                label = label.to(device)

                output = model(image)['out']
                # output is a tensor: the first dimension is the batch size, the second is the number of segmentation classes, the
                # last two are the original image sizes. To obtain a proper annotated prediction, we need to argmax over the second
                # dimension
                output = torch.argmax(output, dim=1)

                # save scores and labels
                start_idx = batch_id * dataloader.batch_size
                end_idx = start_idx + output.size(0)
                golden_scores[start_idx:end_idx] = output
                golden_labels[start_idx:end_idx] = label
        
    return golden_scores, golden_labels


def yolo_detection_golden_run(
        model: YOLO,
        dataloader: torch.utils.data.DataLoader,      
):
    with open('other_nets/detection/coco/cocodetection_ids_to_ultralytics_ids.yaml') as f:
         id_mapping = yaml.load(f, yaml.SafeLoader)

    all_results = []
    all_targets = []

    with torch.no_grad():
        for imgs, target in tqdm(dataloader, desc='Performing golden run', colour='yellow'):
                results = model.predict(imgs)
                all_results.append(results)

                # target is a list of dictionaries, each describing a bounding box. The 'category_id' entry is used as the label
                # for the box, but those ids actually correspond to the original COCO annotations and do not line up with the ids used
                # by Ultralytics, thus causing evaluation errors. This needs to be fixed by mapping to the proper ids.
                for target_item in target:
                    target_item['category_id'] = id_mapping[target_item['category_id']]
                all_targets.append(target)

                print('YOLO is currently doing a single detection iteration for DEBUG purposes. Remember to remove it.')
                break #TODO: remove
        
    return all_results, all_targets



#--ERROR RUNS--
def classification_error_run(
        injected_module: torch.nn.Module,
        error_simulator_pytorch_hook,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_classes: int,
        device,
    ):
    num_samples = len(dataloader) * dataloader.batch_size

    # run inference with injected error and collect results
    with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
        scores = torch.zeros((num_samples, num_classes), device=device)
        
        for batch_id, (image, label) in enumerate(tqdm(dataloader, colour='red')):
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)
        
            # store scores
            start_idx = batch_id * dataloader.batch_size
            end_idx = start_idx + output.size(0)
            scores[start_idx:end_idx, :] = output

    # compute faulty rankings (indices only)
    rankings = scores.topk(num_classes)[1]

    return scores, rankings


def segmentation_error_run(
        injected_module: torch.nn.Module,
        error_simulator_pytorch_hook,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device
    ):
    num_samples = len(dataloader) * dataloader.batch_size
    for sample_img, _ in dataloader: break
    input_shape = sample_img.shape

    # run inference with injected error and collect results
    with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
        scores = torch.zeros((num_samples, input_shape[-2], input_shape[-1]), device=device)
        
        for batch_id, (image, label) in enumerate(tqdm(dataloader, colour='red')):
            image = image.to(device)
            label = label.to(device)
            
            output = model(image)['out']
            output = torch.argmax(output, dim=1)
        
            # store scores
            start_idx = batch_id * dataloader.batch_size
            end_idx = start_idx + output.size(0)
            scores[start_idx:end_idx, :] = output

    return scores


def yolo_detection_error_run(
    injected_module: torch.nn.Module,
    error_simulator_pytorch_hook,
    model: YOLO,
    dataloader: torch.utils.data.DataLoader,
):
    all_results = []

    with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
        for imgs, _ in tqdm(dataloader, colour='red'):
            results = model.predict(imgs)
            all_results.append(results)
    
    return all_results