# Instantiate the encoder and decoder
import torch.nn as nn
import torch.optim as optim
from rock_generator.encoders_decoders.encoder import Encoder
from rock_generator.encoders_decoders.decoder import Decoder


encoder = Encoder()
decoder = Decoder()

# Loss function: L1 Loss
reconstruction_loss_fn = nn.L1Loss()

# Optimizers for the encoder and decoder (use the same optimizer for both or separate)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Training loop example (assuming `spectrograms` is a batch of input log-magnitude spectrograms)
def train_step(spectrograms):
    # Forward pass: Encode and decode the input spectrograms
    latent_space = encoder(spectrograms)
    reconstructed_spectrograms = decoder(latent_space)
    
    # Compute L1 reconstruction loss
    loss = reconstruction_loss_fn(reconstructed_spectrograms, spectrograms)
    
    # Backpropagation and optimization step
    optimizer.zero_grad()  # Zero gradients
    loss.backward()        # Backpropagate the gradients
    optimizer.step()       # Update the model weights
    
    return loss.item()

# Assuming `train_loader` provides batches of spectrogram data
for epoch in range(epochs):
    for batch_spectrograms in train_loader:
        loss = train_step(batch_spectrograms)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')