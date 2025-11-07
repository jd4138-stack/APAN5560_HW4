import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir='checkpoints'):
    """
    Save model checkpoint with:
    - model state dict
    - optimizer state dict
    - epoch number
    - loss and accuracy metrics

    Files are stored under `checkpoint_dir/epoch_{epoch:03d}.pt`
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
        },
        ckpt_path,
    )
    return ckpt_path

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    Load checkpoint and restore training state.

    Returns:
        epoch (int), loss (float), accuracy (float)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", None)
    loss = ckpt.get("loss", None)
    accuracy = ckpt.get("accuracy", None)
    return epoch, loss, accuracy
