from audio_transformation.spectrogram_conversion import *
import matplotlib.pyplot as plt

def display_spectrogram(spectrogram, title="Log-Magnitude Spectrogram"):
    # Remove batch dimension if present
    if len(spectrogram.shape) == 3:
        spectrogram = spectrogram.squeeze(0)

    if spectrogram.shape[0] > 1:
        spectrogram = spectrogram[0, :, :]
    
    # Convert to numpy for visualization
    spectrogram = spectrogram.detach().cpu().numpy()

    # Plot the spectrogram
    plt.figure(figsize=(10, 5))
    plt.imshow(spectrogram, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel('Frequency Bins')
    plt.xlabel('Time Frames')
    plt.show()

def main():
    audio_files = ['D:/Machine Learning/CS 674 Advanced Deep Learning/Project 1 - Music Generation Using GAN/rock_generator/audio_files/The Strange World of A.J. Kaufmann - The Sign.mp3',
                    'D:/Machine Learning/CS 674 Advanced Deep Learning/Project 1 - Music Generation Using GAN/rock_generator/audio_files/PENSAMENTOS RODADOS - bups.mp3']

    spectrogram_batch = process_audio_batch(audio_files)
    for i, spec in enumerate(spectrogram_batch):
        print(f"Spectrogram {i} shape: {spec.shape}")

    dataloader = create_dataloader(spectrogram_batch, batch_size=16, shuffle=True)

    # Iterate through the dataloader
    for batch_idx, batch in enumerate(dataloader):
        # Get the spectrogram (and remove the batch dimension for display)
        spectrogram = batch[0][0]  # Get the first spectrogram in the batch
        
        # Display the spectrogram
        display_spectrogram(spectrogram, title=f"Spectrogram {batch_idx + 1}")
        
        # Optional: Break after displaying the first few spectrograms
        if batch_idx == 2:  # Display 3 spectrograms, for example
            break

if __name__ == "__main__":
    main()