import torch
import torch.utils.data.DataLoader

def move_dataset_to_device(dataloader: torch.utils.data.DataLoader, device):
    """
    Moves the whole image dataset to the device. Returns the list of moved
    images and the list of corresponding labels so they can be iterated over.
    Each item in either list corresponds to an entire batch, whose size is that
    previously specified for the passed DataLoader.
    
    This should NOT be used for large datasets.
    """
    images_gpu = []
    labels = []
    for data in dataloader:
        image, label = data[0].to(device), data[1].to(device)
        images_gpu.append(image)
        labels += label.squeeze().numpy().tolist()
    torch.cuda.synchronize(device)
    
    return images_gpu, labels


def find_corrupted_output_rows(output: torch.Tensor, golden_output: torch.Tensor, tolerance):
    """
    Compares a possibly faulty network output and the corresponding golden output.
    For each element, if abs(output - golden) >= tolerance, a corruption is detected.
    Each row is considered corrupted if at least one of its values is corrupted.
    
    Args
    ----
    * output: network output of shape (batch_size, num_classes)
    * golden_output: golden network output of shape (batch_size, num_classes)
    * tolerance: maximum allowed shift from the golden values
    ----
    Returns
    ----
    * A tensor of shape (batch_size, 1). Each element is boolean; True indicates \
    that the corresponding output row was corrupted. Can be used as a mask \
    to access the faulty output and retrieve only the faulty rows for further processing.
    ----
    """
    error = torch.abs(output - golden_output)
    error_mask = (error >= tolerance).squeeze()
    return torch.all(error_mask, dim=1)


def find_ranking_shifts(faulty_output: torch.Tensor, golden_output: torch.Tensor, topk=5):
    """
    Extracts the topk rankings from the faulty output rows and the corresponding
    golden output rows and compares them.
    
    Returns
    ----
    * A tensor of shape (num_rows, 1). Each element is boolean; True indicates \
    that the rankings are perfectly equal.
    ----
    """
    # extract faulty rankings from output (indices only, sorted)
    faulty_rankings = faulty_output.topk(topk)[1]
    # extract golden rankings (indices only, sorted)
    golden_rankings = golden_output.topk(topk)[1]
    
    equal_rankings = torch.all(torch.eq(faulty_rankings, golden_rankings), -1).squeeze()
    return equal_rankings