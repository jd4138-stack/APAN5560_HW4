# helper_lib/energy/model.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- config (MNIST) ----
IMG_SIZE = 28
CHANNELS = 1

# Swish activation
def swish(x):
    return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
    """
    Conv energy model for 1×28×28 inputs.
    28 -> 14 -> 7 -> 4 -> 2 (via stride-2 convs), then FC to scalar energy.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 16, kernel_size=5, stride=2, padding=2)  # 28->14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)        # 14->7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)        # 7->4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)        # 4->2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)  # (N,1) energy/logit


@torch.no_grad()
def clip_img(x):
    """Map from [-1,1] to [0,1] and clamp."""
    return torch.clamp((x + 1) / 2, 0, 1)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def generate_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """
    Langevin-like sampling in image space using ∇_x E(x).
    inp_imgs in [-1,1], shape (N,1,28,28)
    """
    nn_energy_model.eval()
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)
        inp_imgs.requires_grad_(True)
        energy = nn_energy_model(inp_imgs)            # (N,1)
        grads, = torch.autograd.grad(energy, inp_imgs,
                                     grad_outputs=torch.ones_like(energy))
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
    return inp_imgs.detach()

class Buffer:
    """
    Replay buffer for sampling: mix a few new random images with past samples.
    Keeps up to 8192 images in [-1,1], shape (1,1,28,28) per entry.
    """
    def __init__(self, model, device, capacity=8192, batch_size=128):
        super().__init__()
        self.model = model
        self.device = device
        self.capacity = capacity
        self.batch_size = batch_size
        self.examples = [
            (torch.rand((1, CHANNELS, IMG_SIZE, IMG_SIZE), device=self.device) * 2 - 1)
            for _ in range(batch_size)
        ]

    def sample_new_exmps(self, steps, step_size, noise):
        n_new = np.random.binomial(self.batch_size, 0.05)  # ~5% fresh noise
        new_rand_imgs = torch.rand((n_new, CHANNELS, IMG_SIZE, IMG_SIZE), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.batch_size - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)
        new_imgs = generate_samples(self.model, inp_imgs, steps, step_size, noise)
        # update buffer (prepend)
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[: self.capacity]
        return new_imgs

class Metric:
    def __init__(self):
        self.reset()
    def update(self, val: torch.Tensor):
        self.total += float(val.detach().item())
        self.count += 1
    def result(self):
        return self.total / self.count if self.count else 0.0
    def reset(self):
        self.total = 0.0
        self.count = 0

class EBM(nn.Module):
    """
    Wrapper that owns the EnergyModel + buffer + training step.
    """
    def __init__(self, model: EnergyModel, alpha: float, steps: int, step_size: float, noise: float, device: str):
        super().__init__()
        self.device = device
        self.model = model
        self.buffer = Buffer(self.model, device=self.device)
        self.alpha = alpha
        self.steps = steps
        self.step_size = step_size
        self.noise = noise

        self.loss_metric = Metric()
        self.reg_loss_metric = Metric()
        self.cdiv_loss_metric = Metric()
        self.real_out_metric = Metric()
        self.fake_out_metric = Metric()

    def metrics(self):
        return {
            "loss": self.loss_metric.result(),
            "reg": self.reg_loss_metric.result(),
            "cdiv": self.cdiv_loss_metric.result(),
            "real": self.real_out_metric.result(),
            "fake": self.fake_out_metric.result(),
        }

    def reset_metrics(self):
        for m in [self.loss_metric, self.reg_loss_metric, self.cdiv_loss_metric,
                  self.real_out_metric, self.fake_out_metric]:
            m.reset()

    def train_step(self, real_imgs: torch.Tensor, optimizer: torch.optim.Optimizer):
        # jitter real images
        real_imgs = (real_imgs + torch.randn_like(real_imgs) * self.noise).clamp(-1.0, 1.0)

        # sample fakes from buffer via Langevin steps
        fake_imgs = self.buffer.sample_new_exmps(self.steps, self.step_size, self.noise)

        # concat and score
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).to(self.device)
        inp_imgs.requires_grad_(False)
        out_scores = self.model(inp_imgs)   # (N,1)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        # contrastive divergence + L2 regularizer on scores
        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()

        # metrics
        self.loss_metric.update(loss)
        self.reg_loss_metric.update(reg_loss)
        self.cdiv_loss_metric.update(cdiv_loss)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())

        return self.metrics()

    @torch.no_grad()
    def test_step(self, real_imgs: torch.Tensor):
        bsz = real_imgs.shape[0]
        fake_imgs = torch.rand((bsz, CHANNELS, IMG_SIZE, IMG_SIZE), device=self.device) * 2 - 1
        inp = torch.cat([real_imgs, fake_imgs], dim=0)
        out = self.model(inp)
        real_out, fake_out = torch.split(out, bsz, dim=0)
        cdiv = real_out.mean() - fake_out.mean()
        self.cdiv_loss_metric.update(cdiv)
        self.real_out_metric.update(real_out.mean())
        self.fake_out_metric.update(fake_out.mean())
        return {"cdiv": self.cdiv_loss_metric.result(),
                "real": self.real_out_metric.result(),
                "fake": self.fake_out_metric.result()}
