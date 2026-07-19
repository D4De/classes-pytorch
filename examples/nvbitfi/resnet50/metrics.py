from typing import Iterable
import torch
import torch.nn as nn

from tqdm import tqdm


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def iou(y_true: torch.Tensor, y_pred: torch.Tensor, class_id: int):
    true_mask = y_true == class_id
    pred_mask = y_pred == class_id
    intersect = (true_mask & pred_mask).sum(-1).sum(-1)
    union = (true_mask | pred_mask).sum(-1).sum(-1)
    ious = (intersect / union).nan_to_num(0.0, 0.0, 0.0)
    return ious


def mean_iou(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    classes: Iterable[int],
    return_class_ious=False,
):
    ious = []
    for i in classes:
        ious.append(iou(y_true, y_pred, i))
    ious = torch.cat(ious, dim=-1)
    if return_class_ious:
        return ious.mean(-1), ious
    else:
        return ious.mean(-1)


def validate_classification(
    model: nn.Module,
    valid_dataloader: torch.utils.data.DataLoader,
    n_classes: int,
    epoch: int,
    criterion: nn.Module,
    device,
):
    valid_pbar = tqdm(
        valid_dataloader, desc=f"Epoch {epoch} Validating", colour="green"
    )

    model.eval()

    valid_count = len(valid_dataloader.dataset)

    with torch.no_grad():
        scores = torch.zeros((valid_count, n_classes), device=device)
        labels = torch.zeros((valid_count), device=device, dtype=torch.long)

        for batch_id, (image, label) in enumerate(valid_pbar):
            image = image.to(device)
            label = label.to(device)
            predictions: torch.Tensor = model(image)
            start_idx = batch_id * valid_dataloader.batch_size
            end_idx = start_idx + predictions.size(0)
            scores[start_idx:end_idx, :] = predictions
            labels[start_idx:end_idx] = label

        top_1 = accuracy(scores, labels, k=1)
        top_5 = accuracy(scores, labels, k=5)
        loss = criterion(scores, labels)

    return top_1, top_5, loss
