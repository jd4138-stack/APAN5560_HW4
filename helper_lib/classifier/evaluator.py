# helper_lib/evaluator.py
import torch

@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device: str = "cpu"):
    """
    Evaluate a model on a dataset.

    Args:
        model (nn.Module): Trained model to evaluate.
        data_loader (DataLoader): Data loader for the evaluation set.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        device (str): 'cpu' or 'cuda'.

    Returns:
        (avg_loss, accuracy): Tuple[float, float]
    """
    model = model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        # accumulate weighted by batch size
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        # accuracy
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    avg_loss = running_loss / max(1, total)
    accuracy = correct / max(1, total)

    return avg_loss, accuracy
