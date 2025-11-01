import os
import torch
import numpy as np

from math import sqrt
from enum import Enum
from threading import Thread, Lock
from scipy.stats import kendalltau
from ultralytics.engine.results import Results

from classes.simulators.pytorch.pytorch_fault import PyTorchFault

class ResultType(Enum):
    MASKED: 0
    SDC_SAFE: 1
    SDC_CRITICAL: 2

#--SINGLE METRICS--
def accuracy_topk(scores: torch.Tensor, labels: torch.Tensor, k: int):
    """
    Computes the top-k accuracy given a set of scores and the corresponding true labels.
    
    Args
    ---
    * scores: a tensor of shape (num samples, num classes), containing the resulting class scores for each sample in the dataset/batch
    * labels: a tensor of shape (num samples, ), containing the true label for each sample in the dataset/batch
    * k: the number of ranking positions (from the top) considered when computing the accuracy

    Returns
    ---
    * the accuracy value as a percentage
    """
    num_samples = labels.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(labels.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / num_samples)


def rank_biased_overlap(ranking: torch.Tensor, golden_ranking: torch.Tensor, p: float = 0.62):
    """
    Given a ranking and the corresponding golden ranking, computes the Rank Biased Overlap metric.
    The p value determines how much the top classes in the ranking contribute to the metric.
    See equation @ https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899/
    to determine what value of p to choose.
    By default, we want the first 3 classes to contribute to roughly 90% of the ranking, so we set p to 0.62.
    """
    assert ranking.shape == golden_ranking.shape, f"Rankings have different shapes: {ranking.shape=} and {golden_ranking.shape=}"

    rank_length = ranking.shape[-1]
    sum = 0.0
    # iterate through the ranking
    for d in range(1, rank_length+1):
        ranking_up_to_d = ranking.squeeze()[:d]
        golden_up_to_d = golden_ranking.squeeze()[:d]

        overlap = 0
        for rank_element in ranking_up_to_d:
            overlap += 1 if rank_element in golden_up_to_d else 0

        a_d = overlap/d
        sum += p**d * a_d
    
    return a_d * p**rank_length + ((1-p)/p * sum)


def mIOU(prediction: torch.Tensor, target: torch.Tensor, num_classes: int):
    """
    Computes the mean IOU for a batch of images, given the prediction and the ground truth.
    Additionally, computes the number of true positive, false positive and false negative pixels for each image.
    prediction and target should be of shape (num_samples, height, width)
    """
    assert prediction.shape == target.shape, f'Prediction and target shapes do not match: {prediction.shape=} and {target.shape=}'
    
    num_samples = prediction.shape[0]
    # one row per sample, one column per segmentation class
    ious = torch.zeros((num_samples, num_classes), device=prediction.device)
    # one entry per sample
    TPs = torch.zeros((num_samples,), device=prediction.device)
    FPs = torch.zeros_like(TPs)
    FNs = torch.zeros_like(TPs)

    def _count_nonzero_twice(t: torch.Tensor):
        return torch.count_nonzero(torch.count_nonzero(t, dim=-1), dim=-1)

    for sem_class in range(num_classes):
        pred_inds = (prediction == sem_class) # True where the prediction's pixels are 'sem_class'
        # count number of matching pixels for each entry, shape (num_samples,)
        pred_nums = _count_nonzero_twice(pred_inds)

        target_inds = (target == sem_class) # True where the target's pixels are 'sem_class'
        target_nums = _count_nonzero_twice(target_inds)

        # count how many matching pixels in the target are also in the prediction
        matching = torch.logical_and(pred_inds, target_inds)
        intersection_nums = _count_nonzero_twice(matching)
        # count the total number of pixels
        union_nums = pred_nums + target_nums - intersection_nums

        class_ious = intersection_nums / union_nums

        ious[:, sem_class] = class_ious
        TPs += intersection_nums
        FPs += pred_nums - intersection_nums
        FNs += target_nums - intersection_nums

    # take the average for each row
    mious = torch.mean(ious, dim=-1)
    return mious, TPs, FPs, FNs


def iou_two_bboxes(predicted_box: list[float], golden_box: list[float]):
    """
    The boxes should be two lists with the form [x_min, y_min, x_max, y_max].
    """
    if len(predicted_box) != 4 or len(golden_box) != 4:
        raise ValueError(f'Both boxes should be in the form [x_min, y_min, x_max, y_max], but {len(predicted_box)=} and {len(golden_box)=}')
    
    x1, y1, x2, y2 = predicted_box
    x1g, y1g, x2g, y2g = golden_box

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area


def center_distance_two_bboxes(box1: list[float], box2: list[float]):
    """
    Computes the euclidean distance between the centers of two bounding boxes.
    The boxes should be two lists with the form [x_min, y_min, x_max, y_max].
    """
    if len(box1) != 4 or len(box2) != 4:
        raise ValueError(f'Both boxes should be in the form [x_min, y_min, x_max, y_max], but {len(box1)=} and {len(box2)=}')
    
    x1, y1, x2, y2 = box1
    x1o, y1o, x2o, y2o = box2

    # compute coordinates of the two centers
    xc1 = (x1 + x2) / 2
    yc1 = (y1 + y2) / 2
    xc2 = (x1o + x2o) / 2
    yc2 = (y1o + y2o) / 2

    # compute distance
    distance2 = (xc1 - xc2)**2 + (yc1 - yc2)**2
    return sqrt(distance2)


def yolo_coco_evaluate_golden(ground_truth: list[dict], predictions: Results, iou_threshold=0.5):
    """
    The ground truth is provided by PyTorch's CocoDetection: for a given image, the corresponding targets are packed into a list
    of dictionary; each of these dictionaries corresponds to one annotated object in the image.
    The predictions on the image are the result of running it through Ultralytics's YOLO, which outputs a Results object containing
    a list of detected objects with corresponding bounding boxes.

    This function compares the predictions with the ground truth by computing the IOU metric for each pair of bounding boxes.
    The evaluation outputs the number of true positives, false positives and false negatives.
    """
    TP = 0
    matched_true_indices = [] # true boxes that are matched are subsequently ignored

    predicted_classes, predicted_coords = predictions.boxes.cls.tolist(), predictions.boxes.xyxy.tolist()

    # iterate through the predicted boxes
    for pred_class, pred_xyxy in zip(predicted_classes, predicted_coords):
        max_iou = None
        matched_true_index = None

        # iterate through the true boxes and find the one with maximum iou
        for i, true_result in enumerate(ground_truth):
            if i in matched_true_indices: # skip if the box was matched already
                continue
            
            true_xyxy = true_result['bbox']

            # compute iou
            iou = iou_two_bboxes(pred_xyxy, true_xyxy)

            if max_iou == None or iou > max_iou:
                # iou is the best one so far: update
                max_iou = iou
                matched_true_index = i

        if max_iou != None and max_iou >= iou_threshold: # found a match for the bounding box
            # get the class id of the matched box
            true_id = int(ground_truth[matched_true_index]['category_id'])

            # if the ids match, true positive
            if int(pred_class) == true_id:
                matched_true_indices.append(matched_true_index)
                TP += 1

    FP = len(predicted_classes) - TP # these are the predicted boxes that were not matched
    FN = len(ground_truth) - len(matched_true_indices) # these are the golden boxes that were not matched

    return TP, FP, FN


def yolo_coco_evaluate_corrupted(
    golden: Results, corrupted: Results,
    max_box_distance: float, box_distance_tolerance: float, iou_threshold=0.5
):
    TP = 0
    matched_golden_indices = [] # golden boxes that are matched are subsequently ignored
    mismatched_label_found = False
    # track the matched corrupted box with the highest confidence
    max_corrupted_confidence = None
    max_corrupted_confidence_box_coords = max_corrupted_confidence_golden_box_coords = None

    golden_boxes = golden.boxes
    golden_classes, golden_coords = golden_boxes.cls.tolist(), golden_boxes.xyxy.tolist()

    corrupted_boxes = corrupted.boxes
    corrupted_classes, corrupted_coords, corrupted_conf = corrupted_boxes.cls.tolist(), corrupted_boxes.xyxy.tolist(), corrupted_boxes.conf.tolist()
    
    # first handle cases in which there are no golden boxes or no corrupted boxes
    if not golden_classes and not corrupted_classes: # both are empty: masked
        return 0, 0, 0, ResultType.MASKED

    if not golden_classes or not corrupted_classes: # one is empty, the other is not: critical
        return 0, 0, 0, ResultType.SDC_CRITICAL


    # iterate through the predicted boxes
    for corrupted_class, corrupted_xyxy, corr_conf in zip(corrupted_classes, corrupted_coords, corrupted_conf):
        max_iou = None
        matched_golden_index = None

        # find golden boxes that match the class id
        for i, g_coords in enumerate(golden_coords):
            if i in matched_golden_indices: # skip if the box was matched already
                continue

            # compute iou
            iou = iou_two_bboxes(corrupted_xyxy, g_coords)

            if max_iou == None or iou > max_iou:
                # iou is the best one so far: update
                max_iou = iou
                matched_golden_index = i

        if max_iou != None and max_iou >= iou_threshold: # found a match for the bounding box
            # get class id of the matched box
            matched_id = int(golden_classes[matched_golden_index])

            # if the ids match, true positive
            if int(corrupted_class) == matched_id:
                matched_golden_indices.append(matched_golden_index)
                TP += 1
                # if this is the box with the highest confidence so far, keep track of the coordinates
                if max_corrupted_confidence is None or corr_conf > max_corrupted_confidence:
                    max_corrupted_confidence = corr_conf
                    max_corrupted_confidence_box_coords = corrupted_xyxy
                    max_corrupted_confidence_golden_box_coords = golden_coords[matched_golden_index]

            else:
                # box matches, but label does not
                mismatched_label_found = True
    
    FP = len(corrupted_classes) - TP # these are the predicted boxes that were not matched
    FN = len(golden_classes) - len(matched_golden_indices) # these are the golden boxes that were not matched

    # determine type of result
    result = ResultType.SDC_CRITICAL

    # at this point, we know that there is at least one true box; the result may be non-critical only if at least one
    # box was successfully detected
    if TP > 0:
        # compute the distance between the matched box with the highest confidence score and its golden match
        max_conf_distance = center_distance_two_bboxes(max_corrupted_confidence_box_coords, max_corrupted_confidence_golden_box_coords)
        # normalize the distance
        max_conf_distance = max_conf_distance / max_box_distance

        if TP == len(golden_classes) and not mismatched_label_found: # all boxes match and all labels are correct
            if max_conf_distance < box_distance_tolerance: # distance is close enough to 0
                result = ResultType.MASKED
            elif max_conf_distance < 0.05: # distance is within safe range
                result = ResultType.SDC_SAFE
            # otherwise, result is critical

    return TP, FP, FN, result




#--METRICS COMPUTATION FOR RUNS--
def compute_yolo_detection_golden_run_metrics(golden_results, logger, report_data, runtime):
    predictions, ground_truth = golden_results

    total_precision = total_recall = 0.0
    num_samples = 0

    for batch_prediction, batch_truth in zip(predictions, ground_truth):
        num_samples += len(batch_prediction)

        for single_prediction, single_truth in zip(batch_prediction, batch_truth):
            TP, FP, FN = yolo_coco_evaluate_golden(single_truth, single_prediction)
            total_precision += TP / (TP + FP)
            total_recall += TP / (TP + FN)

    avg_precision = float(total_precision/num_samples)
    avg_recall = float(total_recall/num_samples)

    logger.info(f'Average precision is {avg_precision}, average recall is {avg_recall}')

    report_data['Golden run data'] = {
        'Average precision': avg_precision,
        'Average recall': avg_recall,
        'Golden inference runtime': runtime,
    }

    return None




def compute_classification_run_metrics(
    results, layer_metrics_dict: dict, 
    csv_writer, module_name: str, error_number: int, fault: PyTorchFault,
    tolerance: float, outputs_path: str, compute_single_metrics: bool = False, num_threads: int = 4,
):
    """
    Note: if 'compute_row_metrics' is False, the corrupted rankings will be saved as numpy files for later processing.
    If it is True, single row metrics such as Kendall Tau will be computed immediately, but the process will take much longer.
    """
    golden_scores, golden_labels, golden_rankings, error_scores, error_rankings = results

    # golden metrics
    golden_top1_accuracy = accuracy_topk(golden_scores, golden_labels, 1)
    layer_metrics_dict[module_name][error_number]['Golden Top1 accuracy'] = golden_top1_accuracy


    num_samples = golden_scores.shape[0]

    # compare golden and corrupted results
    erroneous_values_mask: torch.Tensor = (torch.abs(torch.subtract(error_scores, golden_scores)) > tolerance) # True where an element differs from the golden more than tolerance
    corrupted_rows_mask = erroneous_values_mask.any(dim=-1).squeeze() # True for corrupted rows

    num_sdc: int = corrupted_rows_mask.count_nonzero().item()
    num_masked: int = num_samples - num_sdc

    # compare corrupted rankings and corresponding golden ones
    corrupted_rankings: torch.Tensor = error_rankings[corrupted_rows_mask]
    golden_counterpart_rankings: torch.Tensor = golden_rankings[corrupted_rows_mask]
    top1_corrupted = corrupted_rankings[:, 0] # first column = top1 index
    top1_golden = golden_counterpart_rankings[:, 0]

    # check where the top class is different
    different_top1_mask = top1_corrupted.not_equal(top1_golden)

    # get scores corresponding to the top classes
    top1_corrupted_scores = error_scores[corrupted_rows_mask, top1_corrupted]
    top1_golden_scores = golden_scores[corrupted_rows_mask, top1_golden]
    # if the top scores differ by more than 5%, we have a critical SDC
    wrong_scores_mask = (torch.abs(torch.subtract(top1_corrupted_scores, top1_golden_scores)) > 0.05)
    
    # if either of the previous conditions hold, we have a critical SDC, otherwise it's a safe one
    sdc_critical_mask = different_top1_mask.logical_or(wrong_scores_mask)
    num_sdc_critical: int = sdc_critical_mask.count_nonzero().item()
    num_sdc_safe: int = num_sdc - num_sdc_critical

    spatial_pattern = str(fault.spatial_pattern_name)
    if compute_single_metrics:
        compute_classification_row_metrics(
            corrupted_rankings, golden_counterpart_rankings, sdc_critical_mask,
            num_sdc_safe, num_sdc_critical,
            csv_writer, layer_metrics_dict,
            module_name, error_number, spatial_pattern,
            num_threads
        )
    else:
        # store rankings as files
        storing_dir = os.path.join(outputs_path, 'saved_rankings', module_name)
        os.makedirs(storing_dir, exist_ok=True)
        file_prefix = f'err{error_number}_{spatial_pattern}_'
        # safe
        filename = file_prefix + 'sdcsafe_golden.npy'
        np.save(os.path.join(storing_dir, filename), golden_counterpart_rankings[~sdc_critical_mask].cpu().numpy())
        filename = file_prefix + 'sdcsafe_corrupted.npy'
        np.save(os.path.join(storing_dir, filename), corrupted_rankings[~sdc_critical_mask].cpu().numpy())
        # critical
        filename = file_prefix + 'sdccritical_golden.npy'
        np.save(os.path.join(storing_dir, filename), golden_counterpart_rankings[sdc_critical_mask].cpu().numpy())
        filename = file_prefix + 'sdccritical_corrupted.npy'
        np.save(os.path.join(storing_dir, filename), corrupted_rankings[sdc_critical_mask].cpu().numpy())
    
    return num_masked, num_sdc_safe, num_sdc_critical


def compute_classification_row_metrics(
        corrupted_rankings: torch.Tensor, golden_counterpart_rankings: torch.Tensor, sdc_critical_mask: torch.Tensor,
        num_sdc_safe: int, num_sdc_critical: int,
        csv_writer, metrics_dict: dict,
        module_name: str, error_number: int, spatial_pattern_name: str,
        num_threads: int,
):
    # REMINDER: the CSV fields are
    # ['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'Topclass golden', 'Topclass corrupted',
    # 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']
    csv_fields = [module_name, error_number, spatial_pattern_name]

    # set up for parallel computation of single row metrics and csv writing
    thread_lock = Lock()
    thread_metrics_list: list[tuple] = [] # stores final thread metrics as tuples (tau, rbo)

    def _thread_compute_row_metrics(
            corrupted: torch.Tensor, golden: torch.Tensor, csv_fields: list,
            thread_id: int, num_threads: int
    ):
        num_rows = corrupted.shape[0]
        current_row = thread_id

        tau_total = rbo_total = 0.0
        csv_rows = []

        while current_row < num_rows:
            corrupted_row = corrupted[current_row]
            golden_row = golden[current_row]

            tau = kendalltau(corrupted_row, golden_row).statistic.item()
            rbo = rank_biased_overlap(corrupted_row, golden_row)
            tau_total += tau
            rbo_total += rbo

            rest_rankings = str(corrupted_row[1:].tolist()).replace(',', ' |')
            rest_golden_rankings = str(golden_row[1:].tolist()).replace(',', ' |')

            csv_rows.append(csv_fields + [
                golden_row[0].item(), # golden top
                corrupted_row[0].item(), # corrupted top
                tau,
                rbo,
                rest_golden_rankings,
                rest_rankings,
            ])

            current_row += num_threads
        
        # write csv rows and update metrics totals
        thread_lock.acquire()
        csv_writer.writerows(csv_rows)
        thread_metrics_list.append((tau_total, rbo_total))
        thread_lock.release()
            

    tau_safe = tau_critical = rbo_safe = rbo_critical = 0.0
    # critical
    threads: list[Thread] = []

    sdc_critical_rows = corrupted_rankings[sdc_critical_mask].cpu()
    golden_critical_rows = golden_counterpart_rankings[sdc_critical_mask].cpu()
    for thread_id in range(num_threads):
        t = Thread(target = _thread_compute_row_metrics, args=(
            sdc_critical_rows, golden_critical_rows, csv_fields + ['False'], thread_id, num_threads,
        ))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # collect metrics
    for metrics_tuple in thread_metrics_list:
        tau_critical += metrics_tuple[0]
        rbo_critical += metrics_tuple[1]

    # safe
    threads.clear()

    sdc_safe_rows = corrupted_rankings[~sdc_critical_mask].cpu()
    golden_safe_rows = golden_counterpart_rankings[~sdc_critical_mask].cpu()
    for thread_id in range(num_threads):
        t = Thread(target = _thread_compute_row_metrics, args=(
            sdc_safe_rows, golden_safe_rows, csv_fields + ['True'], thread_id, num_threads,
        ))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # collect metrics
    for metrics_tuple in thread_metrics_list:
        tau_safe += metrics_tuple[0]
        rbo_safe += metrics_tuple[1]

    # save average total statistics for this fault
    metrics_dict[module_name][error_number]['Average Kendall Tau (SDC-safe)'] =     (tau_safe / num_sdc_safe) if num_sdc_safe > 0 else 0.0
    metrics_dict[module_name][error_number]['Average Kendall Tau (SDC-critical)'] = (tau_critical / num_sdc_critical) if num_sdc_critical > 0 else 0.0
    metrics_dict[module_name][error_number]['Average RBO (SDC-safe)'] =             (rbo_safe / num_sdc_safe) if num_sdc_safe > 0 else 0.0
    metrics_dict[module_name][error_number]['Average RBO (SDC-critical)'] =         (rbo_critical / num_sdc_critical) if num_sdc_critical > 0 else 0.0




def compute_segmentation_run_metrics(
    results, layer_metrics_dict: dict, 
    csv_writer, module_name: str, error_number: int, fault: PyTorchFault, num_classes: int,
    tolerance: float, outputs_path: str, compute_single_metrics: bool = False, num_threads: int = 4,
):
    # all three of these tensors have shape (num_samples, height, width)
    golden_predictions, ground_truth, error_predictions = results
    num_samples: int = golden_predictions.shape[0]

    # compute average mIOU for the golden run
    golden_run_mious, TP, FP, FN = mIOU(golden_predictions, ground_truth, num_classes)
    avg_golden_miou = torch.mean(golden_run_mious).item()
    avg_golden_precision = torch.mean(TP / (TP + FP)).item()
    avg_golden_recall = torch.mean(TP / (TP + FN)).item()
    layer_metrics_dict[module_name][error_number]['Average golden mIOU'] = avg_golden_miou
    layer_metrics_dict[module_name][error_number]['Average golden precision'] = avg_golden_precision
    layer_metrics_dict[module_name][error_number]['Average golden recall'] = avg_golden_recall

    # compute mIOUs for the error run
    error_run_mious, TP, FP, FN = mIOU(error_predictions, golden_predictions, num_classes)
    reciprocal_error_run_mious = (1.0 - error_run_mious)

    # determine where (1-mIOU) > tolerance: those are SDCs
    sdc_mask = (reciprocal_error_run_mious > tolerance)
    sdc_mious = error_run_mious[sdc_mask]
    TP = TP[sdc_mask]
    FP = FP[sdc_mask]
    FN = FN[sdc_mask]

    num_sdc: int = torch.count_nonzero(sdc_mask).item()
    num_masked: int = num_samples - num_sdc

    # determine where (1-mIOU) > 5%: those are critical SDCs
    sdc_critical_mask = (reciprocal_error_run_mious[sdc_mask] > 0.05)
    num_sdc_critical: int = torch.count_nonzero(sdc_critical_mask).item()
    num_sdc_safe: int = num_sdc - num_sdc_critical

    # add average mIOU, precision and recall to report
    mious_critical = sdc_mious[sdc_critical_mask]
    TP_critical = TP[sdc_critical_mask]
    FP_critical = FP[sdc_critical_mask]
    FN_critical = FN[sdc_critical_mask]
    avg_critical_miou = torch.mean(mious_critical).item()
    avg_critical_precision = torch.mean(TP_critical / (TP_critical + FP_critical)).item()
    avg_critical_recall = torch.mean(TP_critical / (TP_critical + FN_critical)).item()
    layer_metrics_dict[module_name][error_number]['Average SDC critical mIOU'] = avg_critical_miou
    layer_metrics_dict[module_name][error_number]['Average SDC critical precision'] = avg_critical_precision
    layer_metrics_dict[module_name][error_number]['Average SDC critical recall'] = avg_critical_recall

    mious_safe = sdc_mious[~sdc_critical_mask]
    TP_safe = TP[~sdc_critical_mask]
    FP_safe = FP[~sdc_critical_mask]
    FN_safe = FN[~sdc_critical_mask]
    avg_safe_miou = torch.mean(mious_safe).item()
    avg_safe_precision = torch.mean(TP_safe / (TP_safe + FP_safe)).item()
    avg_safe_recall = torch.mean(TP_safe / (TP_safe + FN_safe)).item()
    layer_metrics_dict[module_name][error_number]['Average SDC safe mIOU'] = avg_safe_miou
    layer_metrics_dict[module_name][error_number]['Average SDC safe precision'] = avg_safe_precision
    layer_metrics_dict[module_name][error_number]['Average SDC safe recall'] = avg_safe_recall


    spatial_pattern = str(fault.spatial_pattern_name)
    if compute_single_metrics:
        compute_segmentation_single_metrics(
            sdc_mious, TP, FP, FN, sdc_critical_mask,
            csv_writer, module_name, error_number, spatial_pattern, num_threads)
    else:
        # save SDC tensors (and corresponding golden tensors) to file
        storing_dir = os.path.join(outputs_path, 'saved_outputs', module_name)
        os.makedirs(storing_dir, exist_ok=True)
        sdc_golden_predictions = golden_predictions[sdc_mask]
        sdc_error_predictions = error_predictions[sdc_mask]

        file_prefix = f'err{error_number}_{spatial_pattern}_'
        # safe
        filename = file_prefix + 'sdcsafe_golden.npy'
        np.save(os.path.join(storing_dir, filename), sdc_golden_predictions[~sdc_critical_mask].cpu().numpy())
        filename = file_prefix + 'sdcsafe_corrupted.npy'
        np.save(os.path.join(storing_dir, filename), sdc_error_predictions[~sdc_critical_mask].cpu().numpy())
        # critical
        filename = file_prefix + 'sdccritical_golden.npy'
        np.save(os.path.join(storing_dir, filename), sdc_golden_predictions[sdc_critical_mask].cpu().numpy())
        filename = file_prefix + 'sdccritical_corrupted.npy'
        np.save(os.path.join(storing_dir, filename), sdc_error_predictions[sdc_critical_mask].cpu().numpy())

    return num_masked, num_sdc_safe, num_sdc_critical


def compute_segmentation_single_metrics(
    mious: torch.Tensor, TP: torch.Tensor, FP: torch.Tensor, FN: torch.Tensor, sdc_critical_mask: torch.Tensor,
    csv_writer,
    module_name: str, error_number: int, spatial_pattern_name: str,
    num_threads: int,
):
    # REMINDER: the csv header for segmentation is
    #['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'mIOU', 'Precision', 'Recall']
    csv_fields = [module_name, error_number, spatial_pattern_name]

    # set up for parallel computation of single row metrics and csv writing
    thread_lock = Lock()

    def _thread_write_metrics(
        mious: torch.Tensor, precisions: torch.Tensor, recalls: torch.Tensor, csv_fields: list,
        thread_id: int, num_threads: int
    ):
        num_samples = mious.shape[0]
        current_sample = thread_id

        csv_rows = []

        while current_sample < num_samples:
            csv_rows.append(csv_fields + [
                mious[current_sample].item(),
                precisions[current_sample].item(),
                recalls[current_sample].item(),
            ])

            current_row += num_threads
        
        # write csv rows
        thread_lock.acquire()
        csv_writer.writerows(csv_rows)
        thread_lock.release()
            

    # critical
    threads: list[Thread] = []

    mious_critical = mious[sdc_critical_mask]
    TP_critical = TP[sdc_critical_mask]
    FP_critical = FP[sdc_critical_mask]
    FN_critical = FN[sdc_critical_mask]
    precision_critical = TP_critical / (TP_critical + FP_critical)
    recall_critical = TP_critical / (TP_critical + FN_critical)

    for thread_id in range(num_threads):
        t = Thread(target = _thread_write_metrics, args=(
            mious_critical, precision_critical, recall_critical, csv_fields + ['False'], thread_id, num_threads,
        ))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # safe
    threads.clear()

    mious_safe = mious[~sdc_critical_mask]
    TP_safe = TP[~sdc_critical_mask]
    FP_safe = FP[~sdc_critical_mask]
    FN_safe = FN[~sdc_critical_mask]
    precision_safe = TP_safe / (TP_safe + FP_safe)
    recall_safe = TP_safe / (TP_safe + FN_safe)

    for thread_id in range(num_threads):
        t = Thread(target = _thread_write_metrics, args=(
            mious_safe, precision_safe, recall_safe, csv_fields + ['True'], thread_id, num_threads,
        ))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()




def compute_yolo_detection_run_metrics(
    results, layer_metrics_dict: dict,
    csv_writer, module_name: str, error_number: int, fault: PyTorchFault,
    tolerance: float, outputs_path: str, image_size: int, compute_single_metrics: bool = False, num_threads: int = 4,
):
    """
    Note: tolerance is used as the maximum possible box distance for masking.
    The image size is used to determine the maximum possible box distance, which is then used to normalize the box distances.
    """
    golden_results, targets, error_results = results

    # transform target boxes' coordinates to xyxy
    for target in targets:
        target_xywh = target['bbox']
        # bounding boxes provided by CocoDetection use the form xywh
        target['bbox'] = [
            target_xywh[0], # x_min
            target_xywh[1], # y_min
            target_xywh[0] + target_xywh[2], # x_max
            target_xywh[1] + target_xywh[3], # y_max
        ]

    # the images are assumed to be square, so the maximum possible distance is the length of the diagonal
    max_possible_box_distance = image_size * sqrt(2)

    # Reminder: the csv header is 
    #['Layer name', 'Error number', 'Spatial pattern', 'Safe', 'Precision', 'Recall']
    csv_fields = [module_name, error_number, str(fault.spatial_pattern_name)]
    csv_rows = []

    total_precision = total_recall = 0.0
    num_samples = 0

    # golden metrics
    for batch_prediction, batch_truth in zip(golden_results, targets):
        num_samples += len(batch_prediction)

        for single_prediction, single_truth in zip(batch_prediction, batch_truth):
            TP, FP, FN = yolo_coco_evaluate_golden(single_truth, single_prediction)
            total_precision += TP / (TP + FP)
            total_recall += TP / (TP + FN)

    avg_precision = float(total_precision/num_samples)
    avg_recall = float(total_recall/num_samples)

    layer_metrics_dict[module_name][error_number]['Average golden precision'] = avg_precision
    layer_metrics_dict[module_name][error_number]['Average golden recall'] = avg_recall

    # error metrics
    num_masked = num_sdc_safe = num_sdc_critical = 0
    total_TP, total_FP, total_FN = 0

    for batch_prediction, batch_golden in zip(error_results, golden_results):
        for single_prediction, single_golden in zip(batch_prediction, batch_golden):
            TP, FP, FN, result = yolo_coco_evaluate_corrupted(single_golden, single_prediction, max_possible_box_distance, tolerance)
            
            if result ==  ResultType.MASKED:
                num_masked += 1
            else:
                if result == ResultType.SDC_SAFE: 
                    num_sdc_safe += 1
                    csv_type = 'True'
                elif result == ResultType.SDC_CRITICAL: 
                    num_sdc_critical += 1
                    csv_type = 'False'

                if compute_single_metrics:
                    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
                    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
                    csv_rows.append(csv_fields + [csv_type, precision, recall])
                else:
                    #TODO implement results saving
                    pass

                total_TP += TP
                total_FP += FP
                total_FN += FN

    if compute_single_metrics:
        csv_writer.writerows(csv_rows)

    # save results for the fault
    layer_metrics_dict[module_name][error_number]['Precision'] = total_TP / (total_TP + total_FP) if (total_TP + total_FP) != 0 else 0.0
    layer_metrics_dict[module_name][error_number]['Recall'] = total_TP / (total_TP + total_FN) if (total_TP + total_FN) != 0 else 0.0

    return num_masked, num_sdc_safe, num_sdc_critical