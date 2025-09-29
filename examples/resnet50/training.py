import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import AverageMeter


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * scale
    print(
        "DECAYING learning rate.\n The new LR is %f\n"
        % (optimizer.param_groups[1]["lr"],)
    )


def save_checkpoint(
    epoch,
    model,
    optimizer,
    losses,
    lr_scheduler=None,
    path = None,
    **kwargs
):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        **kwargs,
    }
    # filename = path.format(losses=losses, **state)
    # torch.save(state, filename)
    torch.save(state, path)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def train_one_epoch(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimized_params,
    criterion: nn.Module,
    epoch: int,
    device,
):
    model.train()
    epoch_loss = AverageMeter()
    epoch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training", colour="yellow")
    loss_str = "Loss {loss:.2f}"
    for image, label in epoch_pbar:
        image = image.to(device)
        label = label.to(device)

        predictions = model(image)
        loss = criterion(predictions, label)

        model.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(optimized_params, max_norm=1.5)

        epoch_loss.update(loss)
        epoch_pbar.postfix = loss_str.format(loss=epoch_loss.avg)

    lr_scheduler.step()

    return epoch_loss


