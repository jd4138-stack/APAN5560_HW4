# helper_lib/diffusion/trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from helper_lib.diffusion.model import UNet, DiffusionModel, cosine_diffusion_schedule
from helper_lib.shared.data_loader import get_cifar10_loaders
from helper_lib.shared.checkpoints import save_checkpoint

def train_diffusion(
    image_size: int = 32,                # ↓ 32 is much faster than 64
    batch_size: int = 64,
    epochs: int = 1,
    lr: float = 2e-4,
    device: str | None = None,
    checkpoint_dir: str = "checkpoints/diffusion",
    limit_train_batches: int | None = 200,  # limit work per epoch
    limit_val_batches: int | None = 50,
    use_amp: bool = True,                  # mixed precision on GPU
    num_workers: int = 4,                  # dataloader workers
):
    """
    Fast, controllable Diffusion training loop.
    Saves per-epoch checkpoints and 'best.pt' by lowest val loss.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # PyTorch speedups
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    # Data loaders (make sure your loader resizes to image_size and returns [0,1] tensors)
    train_loader, val_loader, _ = get_cifar10_loaders(
        image_size=image_size, batch_size=batch_size, num_workers=num_workers
    )

    # Model + wrapper
    net = UNet(image_size=image_size, num_channels=3)
    model = DiffusionModel(net, schedule_fn=cosine_diffusion_schedule, ema_decay=0.999).to(device)
    model.set_normalizer(mean=0.5, std=0.5)  # CIFAR-ish normalization

    opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))

    best_val = float("inf")
    best_path = os.path.join(checkpoint_dir, "best.pt")

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Diffusion E{ep}/{epochs}", dynamic_ncols=True, unit="batch")
        for bi, (imgs, _) in enumerate(pbar):
            if limit_train_batches and bi >= limit_train_batches:
                break
            imgs = imgs.to(device)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    loss = model.train_step(imgs, opt, loss_fn)
            else:
                loss = model.train_step(imgs, opt, loss_fn)
            train_losses.append(loss)
            pbar.set_postfix(train=f"{sum(train_losses)/len(train_losses):.4f}")

        # ---- val ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bi, (imgs, _) in enumerate(val_loader):
                if limit_val_batches and bi >= limit_val_batches:
                    break
                imgs = imgs.to(device)
                val_losses.append(model.test_step(imgs, loss_fn))
        val_loss = sum(val_losses) / max(1, len(val_losses))
        avg_train = sum(train_losses) / max(1, len(train_losses))
        print(f"[Epoch {ep}] train={avg_train:.4f}  val={val_loss:.4f}")

        # save epoch ckpt
        path = save_checkpoint(model, opt, epoch=ep, loss=val_loss, accuracy=0.0, checkpoint_dir=checkpoint_dir)
        if not path:  # if your save_checkpoint doesn't return a path
            path = os.path.join(checkpoint_dir, f"epoch_{ep:03d}.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict(), "val_loss": val_loss}, path)

        # track best
        if val_loss < best_val:
            best_val = val_loss
            import shutil
            shutil.copyfile(path, best_path)
            print(f"[best] epoch {ep}  val={best_val:.4f} -> {best_path}")

    return {"model": model, "best_val": best_val, "best_path": best_path}
