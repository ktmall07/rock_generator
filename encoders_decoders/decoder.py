import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=1):
        super(Decoder, self).__init__()

        # Example, update with Decoder Architecture
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # Upsampling by 2
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsampling by 2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),  # Upsampling by 2
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, output_channels, kernel_size=4, stride=2, padding=1),  # Final output (same size as input)
            nn.Tanh()  # To keep the output between -1 and 1 for waveform data
        )

    def forward(self, x):
        return self.decoder(x)
