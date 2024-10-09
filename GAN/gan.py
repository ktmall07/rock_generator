import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, n_fft, n_channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.n_fft = n_fft
        self.n_channels = n_channels

        self.model = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 512, kernel_size=4, stride=1, padding=0), # Initial dense layer
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, n_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Outputs are in the range [-1, 1] for audio
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),  # Assuming 64x64 input
            nn.Sigmoid()  # Binary classification (real or fake)
        )

    def forward(self, x):
        return self.model(x)
