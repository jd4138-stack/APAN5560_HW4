# helper_lib/generator.py
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

@torch.no_grad()
def generate_samples(model, device, num_samples=16, z_dim=100, nrow=4):
    """
    model: a DCGANGenerator (or dict {'G': G, ...})
    Shows a grid of generated MNIST-like digits.
    """
    G = model['G'] if isinstance(model, dict) else model
    G = G.to(device).eval()

    z = torch.randn(num_samples, z_dim, device=device)
    imgs = G(z).cpu()  # (N,1,28,28) in [-1,1]

    grid = make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
    plt.figure(figsize=(nrow, nrow))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.show()
