from rock_generator.encoders_decoders.encoder import Encoder
from rock_generator.encoders_decoders.decoder import Decoder


class Autoencoder(nn.Module):
    def __init__(self, input_channels=1, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction