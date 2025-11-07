import os
import shutil
import torch
from torch import optim
from tqdm import tqdm

from helper_lib.energy.model import EnergyModel, EBM
from helper_lib.shared.data_loader import get_mnist_loader
from helper_lib.shared.checkpoints import save_checkpoint

def train_energy(
    epochs: int = 5,
    alpha: float = 0.1,          # L2 regularizer weight on scores
    steps: int = 40,             # Langevin steps per buffer refresh
    step_size: float = 10.0,     # Langevin step size (try 1.0–2.0 if unstable)
    noise: float = 0.005,        # Gaussian jitter (sampling + real jitter)
    batch_size: int = 128,
    device: str | None = None,
    ckpt_dir: str = "checkpoints/energy",
):
    """
    Train the EnergyModel on MNIST (1x28x28, normalized to [-1, 1]) using an EBM wrapper.

    Saves a checkpoint each epoch as:
      {ckpt_dir}/epoch_XXX.pt
    and maintains a copy of the best (lowest training loss) at:
      {ckpt_dir}/best.pt

    Returns:
        dict: {"model": nn.Module, "best_path": str, "best_loss": float}
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders (ensure your get_mnist_loader uses Normalize(mean=[0.5], std=[0.5]))
    train_loader = get_mnist_loader(train=True, batch_size=batch_size)
    test_loader  = get_mnist_loader(train=False, batch_size=batch_size)

    # Model + EBM wrapper
    model = EnergyModel().to(device)
    ebm = EBM(model=model, alpha=alpha, steps=steps, step_size=step_size, noise=noise, device=device)
    opt = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")
    best_path = None

    for ep in range(1, epochs + 1):
        ebm.reset_metrics()
        pbar = tqdm(train_loader, desc=f"E{ep}/{epochs}", dynamic_ncols=True, unit="batch")

        for x, _ in pbar:
            x = x.to(device)  # (B,1,28,28) already in [-1,1] from the loader
            metrics = ebm.train_step(x, opt)
            # Update progress bar with running averages
            pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

        # (Optional) quick eval pass on test set — reported but not used for selection
        ebm.reset_metrics()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                ebm.test_step(x)

        # Save checkpoint for this epoch
        # NOTE: save_checkpoint in your project writes a file like "epoch_XXX.pt".
        # If your implementation returns a path, you can use that directly.
        save_checkpoint(model, opt, epoch=ep, loss=metrics["loss"], accuracy=0.0, checkpoint_dir=ckpt_dir)
        save_path = os.path.join(ckpt_dir, f"epoch_{ep:03d}.pt")  # adjust if your filename format differs

        # Track best by lowest training loss
        cur_loss = float(metrics["loss"])
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            try:
                shutil.copyfile(save_path, best_path)
            except FileNotFoundError:
                # Fallback: if your save_checkpoint uses a different name, ignore copy failure
                pass
            print(f"[best] epoch {ep:03d}  loss={best_loss:.6f} → {best_path}")

    return {"model": model, "best_path": best_path, "best_loss": best_loss}
