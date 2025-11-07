# scripts/train_gan.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from helper_lib.gan.model import DCGANGenerator, DCGANDiscriminator
from helper_lib.gan.trainer import train_gan
from helper_lib.shared.data_loader import get_mnist_loader
from helper_lib.shared.checkpoints import save_checkpoint

def main(
    device: str | None = None,
    epochs: int = 20,
    batch_size: int = 128,
    z_dim: int = 100,
    lr: float = 2e-4,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # data (MNIST normalized to [-1, 1] to match Tanh)
    train_loader = get_mnist_loader(train=True, batch_size=batch_size)

    # models
    G = DCGANGenerator(z_dim=z_dim)
    D = DCGANDiscriminator()

    # losses/optims (DCGAN defaults)
    criterion = nn.BCEWithLogitsLoss()
    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # train (with tqdm inside)
    out = train_gan(
        model={"G": G, "D": D},
        data_loader=train_loader,
        criterion=criterion,
        optimizer={"G": optim_G, "D": optim_D},
        device=device,
        epochs=epochs,
        z_dim=z_dim,
    )

    # save final generator checkpoint (adjust filename if you like)
    ckpt_dir = Path("checkpoints/gan")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out["G"], optim_G, epoch=epochs, loss=0.0, accuracy=0.0, checkpoint_dir=str(ckpt_dir))

    print(f"[DONE] Saved generator checkpoint(s) under: {ckpt_dir}")
    print("To run the API for GAN:\n  export GAN_G_CKPT=checkpoints/gan/epoch_{:03d}.pt\n".format(epochs))

if __name__ == "__main__":
    # simple CLI: allow overrides via env or edit here
    main()


#-------------bash--------------
# step 1

# export GAN_G_CKPT=checkpoints/gan/epoch_010.pt   # set to actual file
# uv run uvicorn helper_lib.api:app --reload --port 8000

# get png
# step 2

#curl 'http://127.0.0.1:8000/gan/generate?n=16&z_dim=100&nrow=4' \
#| jq -r .image_base64 | base64 --decode > samples.png

