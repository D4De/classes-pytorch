import torch
import torch.utils.data

from model import get_resnet50_model

from metrics import validate_classification
from training import save_checkpoint, train_one_epoch

DEFAULT_DEVICE = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"


def train_resnet50(
    train_dataloader,
    valid_dataloader,
    n_classes=37,
    epochs=50,
    restart_checkpoint_path=None,
    learning_rate=1e-3,
    lr_exp_decay=0.95,
    weight_decay=1e-4,
    device=DEFAULT_DEVICE,
    weight_best_path="./checkpoints/resnet50_train_best.pth",
    weight_latest_path="./checkpoints/resnet50_train_latest.pth",
):
    """
    Loads a ResNet50 model and trains it by using the provided DataLoaders.
    
    The weights of the best run, as well as those of the latest one, are saved
    to the provided paths.
    """
    checkpoint_available = restart_checkpoint_path is not None
    
    model = get_resnet50_model(
        n_classes, pretrained_weights=checkpoint_available, return_transforms=False
    )
    
    # get the model parameters to optimize, e.g. those of the final FC classification layer
    optimized_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        optimized_params, lr=learning_rate, weight_decay=weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, lr_exp_decay
    )
    
    # if a previous training checkpoint is available, load it
    if checkpoint_available:
        print('Found training checkpoint. Loading saved parameters.')
        checkpoint = torch.load(restart_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    criterion = torch.nn.CrossEntropyLoss()

    criterion.to(device)
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            optimized_params,
            criterion,
            epoch,
            device,
        )

        top_1, top_5, loss = validate_classification(
            model, valid_dataloader, n_classes, epoch, criterion, device
        )

        print(f"Validation Top 1 Accuracy: {top_1:.1f}% (Best: {best_accuracy:.1f}%)")
        print(f"Validation Top 5 Accuracy: {top_5:.1f}%")
        print(f"Validation Loss: {loss:.3f}")

        # save a checkpoint if the best accuracy changes
        if top_1 > best_accuracy:
            print("Overwriting best checkpoint")
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                losses=train_loss,
                lr_scheduler=lr_scheduler,
                path=weight_best_path,
            )
            best_accuracy = top_1
        
        # save a checkpoint for the latest epoch
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            losses=train_loss,
            lr_scheduler=lr_scheduler,
            path=weight_latest_path,
        )


