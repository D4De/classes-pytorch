import torch
import numpy as np

from scipy.stats import kendalltau
from ultralytics.engine.results import Results

from classes.simulators.pytorch.pytorch_fault import PyTorchFault

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


def find_corrupted_output_rows(output: torch.Tensor, golden_output: torch.Tensor, tolerance: float = 0.0):
    """
    Compares a possibly faulty network output and the corresponding golden output.
    For each element, if abs(output - golden) >= tolerance, a corruption is detected.
    Each row is considered corrupted if at least one of its values is corrupted.
    
    Args
    ----
    * output: network output of shape (num_samples, num_classes)
    * golden_output: golden network output of shape (num_samples, num_classes)
    * tolerance: maximum allowed shift from the golden values
    ----
    Returns
    ----
    * A tensor of shape (num_samples, 1). Each element is boolean; True indicates \
    that the corresponding output row was corrupted. Can be used as a mask \
    to access the faulty output and retrieve only the faulty rows for further processing.
    ----
    """
    error = torch.abs(output - golden_output) # shape is (num_samples, num_classes), stores absolute element-wise difference
    error_mask = (error >= tolerance).squeeze() # shape is (num_samples, num_classes), stores True where position was corrupted
    return torch.any(error_mask, dim=1) # shape is (num_samples, 1), stores True where row was corrupted


def find_ranking_shifts(faulty_output: torch.Tensor, golden_output: torch.Tensor, topk: int=None):
    """
    Extracts the topk rankings from the faulty output rows and the corresponding
    golden rankings and compares them. If parameter 'topk' is None, the entire ranking is considered.
    
    Returns
    ----
    * A tensor of shape (num_rows, 1). Each element is boolean; True indicates \
    that the rankings are different.
    ----
    """
    if topk is None:
        # use the number of rows to get the entire ranking
        topk = faulty_output.shape[1]

    # extract faulty rankings from output (indices only, sorted)
    faulty_rankings = faulty_output.topk(topk)[1]
    # extract golden rankings (indices only, sorted)
    golden_rankings = golden_output.topk(topk)[1]
    
    equal_rankings = torch.all(torch.eq(faulty_rankings, golden_rankings), -1).squeeze()
    return equal_rankings


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
    Computes the mean IOU for a single image, given the prediction and the ground truth.
    prediction and target should be of size (height, width)
    """
    assert prediction.shape == target.shape, f'Prediction and target shapes do not match: {prediction.shape=} and {target.shape=}'

    #prediction = prediction.max(1)[1].float().cpu().numpy()
    #target = target.float().cpu().numpy() 

    iou_list = list()
    present_iou_list = list()

    for sem_class in range(num_classes):
        pred_inds = (prediction == sem_class)
        target_inds = (target == sem_class)
        if target_inds.sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).sum().item()
            union_now = pred_inds.sum().item() + target_inds.sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    miou = np.mean(present_iou_list).item()
    return miou


def evaluate_segmentation(prediction: torch.Tensor, target: torch.Tensor, num_classes: int):
    """
    Takes an image and the corresponding target (labelled) image and finds the number of true positive, false positive and false negative pixels.
    """
    assert prediction.shape == target.shape, f'Prediction and target shapes do not match: {prediction.shape=} and {target.shape=}'
    TP = FP = FN = 0

    for sem_class in range(num_classes):
        pred_inds = (prediction == sem_class)
        target_inds = (target == sem_class)

        pred_total = pred_inds.sum().item()
        target_total = target_inds.sum().item()

        TP += (pred_inds[target_inds]).sum().item() # pixels that match
        FP += pred_total - TP # predicted pixels that are not actually labelled
        FN += target_total - TP # labelled pixels that were not matched

    return TP, FP, FN


def iou_two_bboxes(predicted_box: list[float], golden_box: list[float]):
    """
    The boxes should be two lists with the form [x_min, y_min, x_max, y_max].
    """
    if len(predicted_box) != 4 or len(golden_box) != 4:
        raise ValueError(f'Both boxes should be in the form [x_min, y_min, x_max, y_max], but {len(predicted_box)=} and {len(golden_box)=}')
    
    def computeArea(coords):
        if coords[2] >= coords[0] and coords[3] >= coords[1]:
            return (coords[2]-coords[0])*(coords[3]-coords[1])
        return 0

    intersection_coords = []
    # x_min
    if predicted_box[0] > golden_box[0]:
        intersection_coords.append(predicted_box[0]) # predicted's left edge is to the right of golden's left side
    else:
        intersection_coords.append(golden_box[0]) # predicted's left edge is to the left of golden's left edge
    # y_min
    if predicted_box[1] > golden_box[1]:
        intersection_coords.append(predicted_box[1]) # predicted's top edge is under golden's top edge
    else:
        intersection_coords.append(golden_box[1]) # predicted's top edge is over golden's top edge
    # x_max
    if predicted_box[2] < golden_box[2]:
        intersection_coords.append(predicted_box[2]) # predicted's right edge is to the left of golden's right edge
    else:
        intersection_coords.append(golden_box[2]) # predicted's right edge is to the right of golden's right edge
    # y_max
    if predicted_box[3] < golden_box[3]:
        intersection_coords.append(predicted_box[3]) # predicted's bottom edge is over golden's bottom edge
    else:
        intersection_coords.append(golden_box[3]) # predicted's bottom edge is under golden's bottom edge

    intersection_area = computeArea(intersection_coords)
    union_area = computeArea(predicted_box) + computeArea(golden_box) - intersection_area
    return float(intersection_area)/union_area


def yolo_coco_evaluate_golden(ground_truth: list[dict], predictions: Results, iou_threshold=0.5):
    """
    The ground truth is provided by PyTorch's CocoDetection: for a given image, the corresponding targets are packed into a list
    of dictionary; each of these dictionaries corresponds to one annotated object in the image.
    The predictions on the image are the result of running it through Ultralytics's YOLO, which outputs a Results object containing
    a list of detected objects with corresponding bounding boxes.

    This function compares the predictions with the ground truth by computing the IOU metric for each pair of bounding boxes.
    The evaluation outputs the number of true positives, false positives and false negatives.
    """
    TP = FP = FN = 0
    matched_true_indices = [] # true boxes that are matched are subsequently ignored

    predicted_classes, predicted_coords = predictions.boxes.cls.tolist(), predictions.boxes.xyxy.tolist()

    # iterate through the predicted boxes
    for pred_class, pred_xyxy in zip(predicted_classes, predicted_coords):
        max_iou = None
        matched_true_index = None

        # find true boxes that match the class id
        for i, true_result in enumerate(ground_truth):
            if i in matched_true_indices: # skip if the box was matched already
                continue

            true_id = int(true_result['category_id'])

            if int(pred_class) == true_id:
                true_xywh = true_result['bbox']
                # bounding boxes provided by CocoDetection use the form xywh: transform to xyxy
                true_xyxy = [
                    true_xywh[0], # x_min
                    true_xywh[1], # y_min
                    true_xywh[0] + true_xywh[2], # x_max
                    true_xywh[1] + true_xywh[3], # y_max
                ]
                # compute iou
                iou = iou_two_bboxes(pred_xyxy, true_xyxy)

                if max_iou == None or iou > max_iou:
                    # iou is the best one so far: update
                    max_iou = iou
                    matched_true_index = i

        if max_iou != None and max_iou >= iou_threshold:
            # found a match for the bounding box
            matched_true_indices.append(matched_true_index)
            TP += 1
    
    FP = len(predicted_classes) - TP # these are the predicted boxes that were not matched
    FN = len(ground_truth) - len(matched_true_indices) # these are the golden boxes that were not matched

    return TP, FP, FN


def yolo_coco_evaluate_corrupted(golden: Results, corrupted: Results, iou_threshold=0.5):
    TP = FP = FN = 0
    matched_golden_indices = [] # golden boxes that are matched are subesequently ignored

    golden_classes, golden_coords = golden.boxes.cls.tolist(), golden.boxes.xyxy.tolist()

    corrupted_boxes = corrupted.boxes
    corrupted_classes, corrupted_coords = corrupted_boxes.cls.tolist(), corrupted_boxes.xyxy.tolist()

    # iterate through the predicted boxes
    for corrupted_class, corrupted_xyxy in zip(corrupted_classes, corrupted_coords):
        max_iou = None
        matched_golden_index = None

        # find golden boxes that match the class id
        for i, golden_class in enumerate(golden_classes):
            if i in matched_golden_indices: # skip if the box was matched already
                continue

            if int(golden_class) == int(corrupted_class):
                golden_xyxy = golden_coords[i]

                # compute iou
                iou = iou_two_bboxes(corrupted_xyxy, golden_xyxy)

                if max_iou == None or iou > max_iou:
                    # iou is the best one so far: update
                    max_iou = iou
                    matched_golden_index = i

        if max_iou != None and max_iou >= iou_threshold:
            # found a match for the bounding box
            matched_golden_indices.append(matched_golden_index)
            TP += 1
    
    FP = len(corrupted_classes) - TP # these are the predicted boxes that were not matched
    FN = len(golden_classes) - len(matched_golden_indices) # these are the golden boxes that were not matched

    return TP, FP, FN




#--METRICS COMPUTATION FOR RUNS--
def compute_classification_golden_run_metrics(golden_results, num_classes: int, logger, report_data, runtime):
    golden_scores, golden_labels = golden_results

     # compute accuracy and other scores
    golden_top1_accuracy = accuracy_topk(golden_scores, golden_labels, 1)
    golden_top5_accuracy = accuracy_topk(golden_scores, golden_labels, 5)
    logger.info(f'top1 accuracy is {golden_top1_accuracy} - top5 accuracy is {golden_top5_accuracy}')

    # get golden rankings (indices only)
    golden_rankings = golden_scores.topk(num_classes)[1]
    
    # save results to report
    # TODO: save other metrics once they're computed
    report_data['Golden run data'] = {
        'Top1 accuracy': golden_top1_accuracy,
        'Top5 accuracy': golden_top5_accuracy,
        'Golden inference runtime': runtime
    }

    return golden_rankings


def compute_segmentation_golden_run_metrics(golden_results, num_classes: int, logger, report_data, runtime):
    predictions, ground_truth = golden_results

    total_miou = 0.0
    num_samples = predictions.shape[0]

    for single_prediction, single_truth in zip(predictions, ground_truth):
        total_miou += mIOU(single_prediction, single_truth, num_classes)
    
    # compute scoring metrics
    average_miou = float(total_miou/num_samples)
    logger.info(f'Average mIOU is {average_miou}')

    report_data['Golden run data'] = {
        'Average mIOU': average_miou,
        'Golden inference runtime': runtime
    }

    return None


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




def compute_classification_final_metrics(
    csv_writer, metrics_dict, module_name: str, error_number: int, fault: PyTorchFault,
    error_results, golden_results,
    tolerance: float, 
):
    scores, rankings = error_results
    golden_scores, golden_labels, golden_rankings = golden_results

    # REMINDER: the CSV fields are
    # ['Layer name', 'Error number', 'Sample index', 'Spatial pattern', 'Topclass golden', 'Topclass corrupted', 'Ranking deviation present',
    # 'Kendall Tau', 'RBO', 'Rest of golden ranking', 'Rest of corrupted ranking']
    num_masked = num_sdc_safe = num_sdc_critical = 0
    total_tau = total_rbo = 0.0

    # compare row by row
    for i, output_row in enumerate(scores):
        csv_fields = [module_name, error_number]
        golden_row = golden_scores[i]

        error = torch.abs(output_row - golden_row) # subtract the two rows
        error = (error >= tolerance).squeeze() # find where the difference is greater than the tolerance

        if error.any().item(): # if at any point the difference is greater than the tolerance, the row is corrupted
            csv_fields.append(i) # add sample index
            csv_fields.append(fault.spatial_pattern_name) # add spatial pattern

            rankings_row = rankings[i]
            golden_rankings_row = golden_rankings[i]

            csv_fields.append(golden_rankings_row[0].item()) # add golden topclass
            csv_fields.append(rankings_row[0].item()) # add corrupted topclass

            # check if all ranking values match
            rankings_are_equal = torch.all(torch.eq(rankings_row, golden_rankings_row), -1).squeeze().item()
            if rankings_are_equal:
                # SDC SAFE
                num_sdc_safe += 1
                ranking_deviation_field = 'No'
                tau = 1.0
                rbo = 1.0
            else:
                # SDC CRITICAL
                num_sdc_critical += 1
                ranking_deviation_field = 'Yes'
                tau = kendalltau(rankings_row.cpu(), golden_rankings_row.cpu()).statistic.item()
                rbo = rank_biased_overlap(rankings_row, golden_rankings_row)

            csv_fields.append(ranking_deviation_field)
            csv_fields.append(tau)
            csv_fields.append(rbo)

            total_tau += tau
            total_rbo += rbo

            # get the remaining parts of the two rankings
            rest_rankings = str(rankings_row[1:].tolist()).replace(',', ' |')
            rest_golden_rankings = str(golden_rankings_row[1:].tolist()).replace(',', ' |')
            csv_fields.append(rest_golden_rankings)
            csv_fields.append(rest_rankings)
            csv_writer.writerow(csv_fields)
        else:
            # MASKED
            num_masked += 1
        
        # save average total statistics for this fault
        num_samples = scores.shape[0]
        metrics_dict[module_name][error_number]['Average Kendall Tau'] = total_tau / num_samples
        metrics_dict[module_name][error_number]['Average RBO'] = total_rbo / num_samples
    
    return num_masked, num_sdc_safe, num_sdc_critical


def compute_segmentation_final_metrics(
    csv_writer, metrics_dict, module_name: str, error_number: int, fault: PyTorchFault,
    error_results, golden_results,
    num_classes: int, tolerance: float = 0.5,
):
    """
    Note: tolerance is used as a mIOU threshold to determine if a corruption is SDC safe or critical.
    """
    error_predictions = error_results
    golden_predictions, ground_truth, other_info = golden_results

    num_masked = num_sdc_safe = num_sdc_critical = 0
    total_TP, total_FP, total_FN = 0

    current_index = 0

    for error_prediction, golden_prediction in zip(error_predictions, golden_predictions):
        if (error_prediction != golden_prediction).any().item(): # if at any point the pixel prediction is different, the output was corrupted
            miou = mIOU(error_prediction, golden_prediction, num_classes)
            if miou > tolerance:
                num_sdc_critical += 1
            else:
                num_sdc_safe += 1

            TP, FP, FN = evaluate_segmentation(error_prediction, golden_prediction, num_classes)

            csv_fields = [module_name, error_number]
            csv_fields.append(current_index) # add sample index
            csv_fields.append(fault.spatial_pattern_name) # add spatial pattern
            csv_fields.append(miou)
            csv_fields.append(TP / (TP + FP)) # precision
            csv_fields.append(TP / (TP + FN)) # recall
            csv_writer.writerow(csv_fields)

            total_TP += TP
            total_FP += FP
            total_FN += FN

            current_index += 1
        else:
            # MASKED
            num_masked += 1

    # save metrics for the fault
    metrics_dict[module_name][error_number]['Precision'] = total_TP / (total_TP + total_FP)
    metrics_dict[module_name][error_number]['Recall'] = total_TP / (total_TP + total_FN)

    return num_masked, num_sdc_safe, num_sdc_critical


def compute_yolo_detection_final_metrics(
    csv_writer, metrics_dict, module_name: str, error_number: int, fault: PyTorchFault,
    error_results, golden_results,
    tolerance: float=0.5,
):
    """
    Note: tolerance is used as the IOU threshold.
    """
    error_predictions = error_results
    golden_predictions, ground_truth, other_info = golden_results

    num_masked = num_sdc_safe = num_sdc_critical = 0
    total_TP, total_FP, total_FN = 0

    current_index = 0

    for batch_prediction, batch_golden in zip(error_predictions, golden_predictions):
        for single_prediction, single_golden in zip(batch_prediction, batch_golden):
            TP, FP, FN = yolo_coco_evaluate_corrupted(single_golden, single_prediction, tolerance)
            
            if TP == len(single_golden):
                # all bounding boxes were matched closely enough: consider this masked
                num_masked += 1
            else:
                if TP > tolerance * len(single_golden):
                    # a fraction of at least 'tolerance' of the boxes is correct: consider safe
                    num_sdc_safe += 1
                else:
                    num_sdc_critical += 1

                csv_fields = [module_name, error_number]
                csv_fields.append(current_index)
                csv_fields.append(fault.spatial_pattern_name)
                csv_fields.append(TP / (TP + FP)) # precision
                csv_fields.append(TP / (TP + FN)) # recall
                csv_writer.writerow(csv_fields)

                total_TP += TP
                total_FP += FP
                total_FN += FN

                current_index += 1

    # save results for the fault
    metrics_dict[module_name][error_number]['Precision'] = total_TP / (total_TP + total_FP)
    metrics_dict[module_name][error_number]['Recall'] = total_TP / (total_TP + total_FN)

    return num_masked, num_sdc_safe, num_sdc_critical