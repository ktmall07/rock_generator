import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(Encoder, self).__init__()
       
        # 1D Conv layers for downsampling the waveform (encoder)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=4, stride=2, padding=1), # Downsampling by 2
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),  # Downsampling by 2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # Downsampling by 2
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, kernel_size=4, stride=2, padding=1),  # Latent representation
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)