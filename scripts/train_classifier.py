# scripts/train_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from helper_lib.classifier.model import get_model
from helper_lib.classifier.trainer import train_model
from helper_lib.classifier.evaluator import evaluate_model
from helper_lib.shared.data_loader import get_cifar10_loaders
from helper_lib.shared.checkpoints import save_checkpoint

def main(
    device: str | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 loaders; images resized to 64x64 elsewhere if your CNN expects it
    train_loader, val_loader, test_loader = get_cifar10_loaders(image_size=64, batch_size=batch_size)

    model = get_model("CNN", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = train_model(
        model, train_loader, val_loader,
        criterion, optimizer,
        device=device, epochs=epochs,
        checkpoint_dir="checkpoints/classifier"
    )

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device=device)
    print(f"[TEST] Loss={test_loss:.4f} | Acc={test_acc*100:.2f}%")

    # (Optional) save once more as a canonical name
    Path("checkpoints/classifier").mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, epoch=epochs, loss=test_loss, accuracy=test_acc,
                    checkpoint_dir="checkpoints/classifier")
    print("To use in API:\n  export CLASSIFIER_CKPT=checkpoints/classifier/best.pt (or your epoch file)")

if __name__ == "__main__":
    main()

