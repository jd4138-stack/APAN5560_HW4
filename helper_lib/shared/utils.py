# helper_lib/utils.py
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn

# ---------------------------
# Reproducibility
# ---------------------------
def seed_everything(seed: int = 42):
    """Set seeds for Python, NumPy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int):
    """
    Seed DataLoader workers deterministically.
    Use: DataLoader(..., worker_init_fn=worker_init_fn)
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

def make_generator(seed: int = 42) -> torch.Generator:
    """
    Optional: pass to DataLoader(generator=...) for reproducible shuffling.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g

# ---------------------------
# Device & model helpers
# ---------------------------
def get_device(prefer: str = "cuda") -> torch.device:
    """
    Pick a device. prefer='cuda' or 'cpu'.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def infer_flat_features(model: nn.Module, input_shape=(1, 3, 64, 64)) -> int:
    """
    Run a dummy forward through the feature extractor to get flattened size.
    Useful when wiring Linear layers dynamically.
    """
    model = model.eval()
    with torch.no_grad():
        x = torch.zeros(*input_shape)
        out = model.features(x) if hasattr(model, "features") else model(x)
        return int(torch.flatten(out, 1).shape[1])

# ---------------------------
# Metrics
# ---------------------------
def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    """
    Compute top-k accuracies. Returns a list of floats in [0,1].
    """
    with torch.no_grad():
        maxk = max(topk)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k / targets.size(0)).item())
        return res if len(res) > 1 else res[0]

class AverageMeter:
    """
    Tracks and averages a metric over time.
    Example:
        loss_meter = AverageMeter()
        for loss in losses: loss_meter.update(loss, n=batch_size)
        print(loss_meter.avg)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)

# ---------------------------
# Logging / persistence
# ---------------------------
def save_history_csv(history, path: str):
    """
    Save a list of dicts (e.g., per-epoch metrics) to CSV.
    Example row keys: epoch, loss, acc, val_loss, val_acc
    """
    if not history:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted({k for row in history for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
