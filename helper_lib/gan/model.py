import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=True),
            nn.Tanh(),
        )
    def forward(self, z):
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.net(x)

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)
    def forward(self, x):
        h = self.features(x)
        return self.classifier(h.view(h.size(0), -1))
