import yaml
import torch

from tqdm import tqdm
from ultralytics import YOLO

from classes.simulators.pytorch.simulator_hook import applied_hook


def classification_run(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device,
    num_classes: int,
    injected_module: torch.nn.Module,
    error_simulator_pytorch_hook,
    use_single_batch=True,
):
    if use_single_batch:
         # get first batch from dataloader
        for imgs, labels in dataloader: break
        imgs = imgs.to(device)
        golden_labels = labels.to(device)
        
        with torch.no_grad():
            # golden inference
            golden_scores = model(imgs).to(device)

            # error inference
            with applied_hook(injected_module, error_simulator_pytorch_hook):
                error_scores = model(imgs).to(device)

    else:
        # perform inference on entire dataset
        num_samples = len(dataloader) * dataloader.batch_size

        # prepare a golden score matrix: one row per sample, one column per class score
        golden_scores = torch.zeros((num_samples, num_classes), device=device)
        # prepare a golden label matrix: one row per sample, one column for the golden label
        golden_labels = torch.zeros((num_samples), device=device)
        # prepare an error score matrix, same shape as golden_scores
        error_scores = torch.zeros_like(golden_scores, device=device)

        # golden run
        with torch.no_grad():
            for batch_id, (imgs, label) in enumerate(tqdm(dataloader, desc='Performing golden run', colour='yellow')):
                imgs = imgs.to(device)
                label = label.to(device)

                output = model(imgs).to(device)
                # IMPORTANT: image is a batch of samples, so output can be considered a matrix with one row per sample in the batch
                # and 'num_classes' columns listing the probabilities for that sample to correspond to each class

                # save scores and labels
                start_idx = batch_id * dataloader.batch_size
                end_idx = start_idx + output.size(0)
                golden_scores[start_idx:end_idx, :] = output
                golden_labels[start_idx:end_idx] = label
            
        # error run
        # run inference with injected error and collect results
        with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
            for batch_id, (imgs, _) in enumerate(tqdm(dataloader, colour='red')):
                imgs = imgs.to(device)
                output = model(imgs).to(device)
            
                # store scores
                start_idx = batch_id * dataloader.batch_size
                end_idx = start_idx + output.size(0)
                error_scores[start_idx:end_idx, :] = output

    # compute result rankings (indices only)
    golden_rankings = golden_scores.topk(num_classes)[1]
    error_rankings = error_scores.topk(num_classes)[1]

    return golden_scores, golden_labels, golden_rankings, error_scores, error_rankings


def segmentation_run(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device,
    injected_module: torch.nn.Module,
    error_simulator_pytorch_hook,
    use_single_batch=True,
):
    if use_single_batch:
        num_samples = dataloader.batch_size
        # get batch
        for imgs, ground_truth in dataloader: break
        imgs = imgs.to(device)
        ground_truth = ground_truth.to(device)

        with torch.no_grad():
            # golden inference
            output = model(imgs)['out']
            golden_out = torch.argmax(output, dim=1).to(device)

        # error inference
        with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
            output = model(imgs)['out']
            error_out = torch.argmax(output, dim=1).to(device)

    else:
        num_samples = len(dataloader) * dataloader.batch_size
        # get shape of the images
        for sample_img, _ in dataloader: break
        input_shape = sample_img.shape

        # prepare a golden out matrix: one tensor for each sample in the dataloader, each tensor has the same shape as the sample
        golden_out = torch.zeros((num_samples, input_shape[-2], input_shape[-1]), device=device)
        # prepare a ground truth matrix, same shape as the golden output
        ground_truth = torch.zeros_like(golden_out, device=device)
        # prepare an error out matrix, same shape as the golden output
        error_out = torch.zeros_like(golden_out, device=device)

        with torch.no_grad():
            for batch_id, (imgs, truth) in enumerate(tqdm(dataloader, desc='Performing golden run', colour='yellow')):
                imgs = imgs.to(device)
                truth = truth.to(device)

                output = model(imgs)['out']
                # output is a tensor: the first dimension is the batch size, the second is the number of segmentation classes, the
                # last two are the original image sizes. To obtain a proper annotated prediction, we need to argmax over the second
                # dimension
                output = torch.argmax(output, dim=1).to(device)

                # save golden output and ground truth
                start_idx = batch_id * dataloader.batch_size
                end_idx = start_idx + output.size(0)
                golden_out[start_idx:end_idx] = output
                ground_truth[start_idx:end_idx] = truth

        # run inference with injected error and collect results
        with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
            for batch_id, (imgs, _) in enumerate(tqdm(dataloader, colour='red')):
                imgs = imgs.to(device)
                output = model(imgs)['out']
                output = torch.argmax(output, dim=1).to(device)
            
                # store scores
                start_idx = batch_id * dataloader.batch_size
                end_idx = start_idx + output.size(0)
                error_out[start_idx:end_idx, :] = output

    return golden_out, ground_truth, error_out


def yolo_detection_run(
    model: YOLO,
    dataloader: torch.utils.data.DataLoader,
    injected_module: torch.nn.Module,
    error_simulator_pytorch_hook,
    use_single_batch=True,
    image_size=128,
    batch_size=64,
):
    with open('other_nets/detection/coco/cocodetection_ids_to_ultralytics_ids.yaml') as f:
        id_mapping = yaml.load(f, yaml.SafeLoader)

    if use_single_batch:
        for imgs, targets in dataloader: break
        with torch.no_grad():
            results = model.predict(imgs, imgsz=image_size, device=0, batch=len(imgs), verbose=False)

        # target is a list of dictionaries, each describing a bounding box. The 'category_id' entry is used as the label
        # for the box, but those ids actually correspond to the original COCO annotations and do not line up with the ids used
        # by Ultralytics, thus causing evaluation errors. This needs to be fixed by mapping to the proper ids.
        for target_item in targets:
            target_item['category_id'] = id_mapping[target_item['category_id']]

        with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
            error_results = model.predict(imgs, imgsz=image_size, device=0, batch=len(imgs), verbose=False)

    else:
        golden_results = []
        targets = []
        error_results = []

        with torch.no_grad():
            for imgs, target in tqdm(dataloader, desc='Performing golden run', colour='yellow'):
                results = model.predict(imgs, imgsz=image_size, device=0, batch=batch_size, verbose=False)
                golden_results.append(results)

                for target_item in target:
                    target_item['category_id'] = id_mapping[target_item['category_id']]
                targets.append(target)

        with applied_hook(injected_module, error_simulator_pytorch_hook), torch.no_grad():
            for imgs, _ in tqdm(dataloader, colour='red'):
                results = model.predict(imgs, imgsz=image_size, device=0, batch=batch_size, verbose=False)
                error_results.append(results)
    
    return golden_results, targets, error_results


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