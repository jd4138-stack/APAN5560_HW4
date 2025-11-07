import os
import torch
from tqdm import tqdm
from helper_lib.shared.checkpoints import save_checkpoint
from helper_lib.classifier.evaluator import evaluate_model

def _epoch_pass(model, loader, criterion, optimizer=None, device="cpu"):
    """
    Runs one pass over a loader. If optimizer is provided, trains; otherwise validates.
    Returns (avg_loss, accuracy).
    """
    is_train = optimizer is not None
    model.train(mode=is_train)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

def train_model(model, train_loader, val_loader, criterion, optimizer, device='cpu', epochs=10, checkpoint_dir='checkpoints'):
    """
    Enhanced training loop with checkpoint functionality

    Implements:
    1) Multi-epoch training
    2) Tracks train/val loss & accuracy
    3) Saves a checkpoint each epoch
    4) Saves the best-performing model (by val accuracy) at {checkpoint_dir}/best.pt
    5) Returns the trained model with BEST weights loaded
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        train_loss, train_acc = _epoch_pass(model, train_loader, criterion, optimizer, device=device)

        # ---- Validate ----
        with torch.no_grad():
            val_loss, val_acc = _epoch_pass(model, val_loader, criterion, optimizer=None, device=device)

        # ---- Logging ----
        print(f"[Epoch {epoch:03d}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # ---- Always save an epoch checkpoint (use val metrics for monitoring) ----
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            accuracy=val_acc,
            checkpoint_dir=checkpoint_dir,
        )

        # ---- Track & save best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(
                {
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": val_loss,
                    "accuracy": val_acc,
                },
                best_path,
            )

    # ---- Load best weights back into the model before returning ----
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10, z_dim=100):
    """
    model: {'G': generator, 'D': discriminator} or (G, D)
    optimizer: {'G': optim_G, 'D': optim_D}
    criterion: BCEWithLogitsLoss
    Shows tqdm progress bars for epochs and batches.
    """
    # Unpack inputs
    if isinstance(model, (tuple, list)):
        G, D = model
    else:
        G, D = model['G'], model['D']
    optim_G = optimizer['G']
    optim_D = optimizer['D']

    G.to(device).train()
    D.to(device).train()

    for epoch in range(1, epochs + 1):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        n_seen = 0

        # tqdm bar for batches in this epoch
        pbar = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for real, _ in pbar:
            real = real.to(device)
            bsz = real.size(0)
            n_seen += bsz

            # -----------------------
            # Train Discriminator
            # -----------------------
            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z).detach()

            real_lbl = torch.ones(bsz, 1, device=device)
            fake_lbl = torch.zeros(bsz, 1, device=device)

            D_real = D(real)
            D_fake = D(fake)

            loss_D_real = criterion(D_real, real_lbl)
            loss_D_fake = criterion(D_fake, fake_lbl)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            optim_D.zero_grad(set_to_none=True)
            loss_D.backward()
            optim_D.step()

            # -----------------------
            # Train Generator
            # -----------------------
            z = torch.randn(bsz, z_dim, device=device)
            gen = G(z)
            D_gen = D(gen)
            loss_G = criterion(D_gen, real_lbl)  # want D to label fakes as real

            optim_G.zero_grad(set_to_none=True)
            loss_G.backward()
            optim_G.step()

            d_loss_epoch += loss_D.item() * bsz
            g_loss_epoch += loss_G.item() * bsz

            # Update tqdm postfix with running averages
            pbar.set_postfix({
                "D_loss": f"{d_loss_epoch / n_seen:.4f}",
                "G_loss": f"{g_loss_epoch / n_seen:.4f}"
            })

        print(f"[Epoch {epoch:03d}] D_loss={(d_loss_epoch/n_seen):.4f} | G_loss={(g_loss_epoch/n_seen):.4f}")

    return {'G': G, 'D': D}