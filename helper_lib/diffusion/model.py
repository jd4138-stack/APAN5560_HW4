# helper_lib/diffusion/model.py
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Time embedding
# ---------------------------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies: int = 16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, t: torch.Tensor):
        """
        t: (B,1,1,1) in [0,1]
        returns: (B, 1, 1, 2*num_frequencies)
        """
        x = t.expand(-1, 1, 1, self.num_frequencies)
        sin = torch.sin(self.angular_speeds * x)
        cos = torch.cos(self.angular_speeds * x)
        return torch.cat([sin, cos], dim=-1)

# ---------------------------
# Diffusion schedules
# ---------------------------
def linear_diffusion_schedule(diffusion_times: torch.Tensor, min_rate=1e-4, max_rate=0.02):
    """
    diffusion_times: (B,1,1,1) in [0,1]
    Returns (noise_rates, signal_rates), both shaped like (B,1,1,1)
    """
    dt = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + dt * (max_rate - min_rate)
    alphas = 1.0 - betas
    # use cumulative product by mapping times to steps in [0,1]—approximation for simplicity
    # For per-sample times we pretend alphas are constant over time step of size 1.
    alpha_bars = torch.cumprod(alphas, dim=0) if dt.dim() == 1 else alphas  # fallback
    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates

def cosine_diffusion_schedule(diffusion_times: torch.Tensor):
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates  = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates

def offset_cosine_diffusion_schedule(diffusion_times: torch.Tensor, min_signal_rate=0.02, max_signal_rate=0.95):
    orig = diffusion_times.shape
    dt = diffusion_times.flatten()
    start = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=dt.device))
    end   = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=dt.device))
    angles = start + dt * (end - start)
    signal_rates = torch.cos(angles).reshape(orig)
    noise_rates  = torch.sin(angles).reshape(orig)
    return noise_rates, signal_rates

# ---------------------------
# UNet-ish backbone
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_ch, affine=False)
        self.norm2 = nn.BatchNorm2d(out_ch, affine=False)

    @staticmethod
    def swish(x): return x * torch.sigmoid(x)

    def forward(self, x):
        res = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + res

class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_ch):
        super().__init__()
        blocks = []
        ch = in_ch
        for _ in range(block_depth):
            blocks.append(ResidualBlock(ch, width))
            ch = width
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, skips):
        for b in self.blocks:
            x = b(x)
            skips.append(x)
        x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_ch):
        super().__init__()
        blocks = []
        ch = in_ch
        for _ in range(block_depth):
            blocks.append(ResidualBlock(ch + width, width))
            ch = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for b in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = b(x)
        return x

class UNet(nn.Module):
    """
    UNet that predicts noise ε(x_t, t).
    """
    def __init__(self, image_size=64, num_channels=3, emb_dim=32):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.embed = SinusoidalEmbedding(num_frequencies=emb_dim // 2)

        self.initial = nn.Conv2d(num_channels, 32, 1)
        self.embed_proj = nn.Conv2d(emb_dim, 32, 1)

        self.down1 = DownBlock(32, block_depth=2, in_ch=32 + 32)
        self.down2 = DownBlock(64, block_depth=2, in_ch=32)
        self.down3 = DownBlock(96, block_depth=2, in_ch=64)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1  = UpBlock(96, block_depth=2, in_ch=128)
        self.up2  = UpBlock(64, block_depth=2, in_ch=96)
        self.up3  = UpBlock(32, block_depth=2, in_ch=64)

        self.final = nn.Conv2d(32, num_channels, 1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        """
        noisy_images: (B,C,H,W)
        noise_variances: (B,1,1,1) in [0,1], we embed and tile to H×W
        """
        skips = []
        x = self.initial(noisy_images)
        emb = self.embed(noise_variances)                 # (B,1,1,emb_dim)
        emb = F.interpolate(emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, self.embed_proj(emb)], dim=1)

        x = self.down1(x, skips)   # -> 32
        x = self.down2(x, skips)   # -> 64
        x = self.down3(x, skips)   # -> 96

        x = self.mid1(x)
        x = self.mid2(x)

        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)

        return self.final(x)       # predicted noise ε

# ---------------------------
# Diffusion wrapper
# ---------------------------
class DiffusionModel(nn.Module):
    """
    Wraps a UNet + schedule; trains to predict noise; samples via a simple reverse loop.
    """
    def __init__(self, model: UNet, schedule_fn=cosine_diffusion_schedule, ema_decay=0.999):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = ema_decay
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.5
        self.normalizer_std  = 0.5

    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std  = std

    def _denoise_step(self, noisy_images, noise_rates, signal_rates, training: bool):
        net = self.network if training else self.ema_network
        if training: net.train()
        else:        net.eval()
        pred_noises = net(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / (signal_rates + 1e-8)
        return pred_noises, pred_images

    @torch.no_grad()
    def reverse_diffusion(self, initial_noise: torch.Tensor, diffusion_steps: int):
        # t goes 1 -> 0
        step = 1.0 / diffusion_steps
        x = initial_noise
        for i in range(diffusion_steps):
            t = torch.ones((x.size(0), 1, 1, 1), device=x.device) * (1.0 - i * step)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self._denoise_step(x, noise_rates, signal_rates, training=False)
            # simple Euler update to next time
            t_next = t - step
            next_noise, next_signal = self.schedule_fn(t_next)
            x = next_signal * pred_images + next_noise * pred_noises
        return pred_images

    @torch.no_grad()
    def generate(self, num_images: int, diffusion_steps: int, image_size=64, initial_noise=None):
        device = next(self.parameters()).device
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=device)
        imgs = self.reverse_diffusion(initial_noise, diffusion_steps)
        # map back to [0,1] using stored normalizer
        return torch.clamp(imgs * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def train_step(self, images: torch.Tensor, optimizer: torch.optim.Optimizer, loss_fn):
        # normalize to roughly zero-mean unit-ish std for CIFAR10
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        # sample per-sample times in [0,1]
        t = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(t)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self._denoise_step(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # EMA
        with torch.no_grad():
            for ema_p, p in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_p.copy_(self.ema_decay * ema_p + (1. - self.ema_decay) * p)

        return float(loss.item())

    @torch.no_grad()
    def test_step(self, images: torch.Tensor, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        t = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(t)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, _ = self._denoise_step(noisy_images, noise_rates, signal_rates, training=False)
        loss = loss_fn(pred_noises, noises)
        return float(loss.item())
