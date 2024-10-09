# Instantiate the encoder and decoder
import torch
import torch.nn as nn
import torch.optim as optim
from rock_generator.autoencoder.encoder import Encoder
from rock_generator.autoencoder.decoder import Decoder

# Training loop example (assuming `spectrograms` is a batch of input log-magnitude spectrograms)
def train_ae(spectrograms):
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

def train_gd(generator, discriminator, autoencoder, dataloader, optimizer_g, optimizer_d, epochs, latent_dim):
    for epoch in range(epochs):
        for real_audio in dataloader:
            batch_size = real_audio.size(0)
            
            # Train Discriminator
            optimizer_d.zero_grad()
            # Generate noise and pass through the generator
            z = torch.randn(batch_size, latent_dim, 1)  # Random latent vector
            fake_audio = generator(z)
            
            # Get discriminator predictions
            real_preds = discriminator(real_audio)
            fake_preds = discriminator(fake_audio.detach())
            
            # Calculate loss
            d_loss = (adversarial_loss_fn(real_preds, torch.ones_like(real_preds)) + 
                       adversarial_loss_fn(fake_preds, torch.zeros_like(fake_preds))) / 2
            
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            fake_preds = discriminator(fake_audio)
            g_loss = loss_function(fake_preds, torch.ones_like(fake_preds))  # Fool the discriminator
            
            g_loss.backward()
            optimizer_g.step()

            print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

encoder = Encoder()
decoder = Decoder()

# Loss function: L1 Loss
reconstruction_loss_fn = nn.L1Loss()
adversarial_loss_fn = nn.BCELoss()

# Optimizers for the encoder and decoder (use the same optimizer for both or separate)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# Assuming `train_loader` provides batches of spectrogram data
for epoch in range(epochs):
    for batch_spectrograms in train_loader:
        loss = train_ae(batch_spectrograms)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}')
