from audio_transformation.spectrogram_conversion import *
from utils.utils import *

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
    
        reverted_spectrogram = revert_spectrogram(spectrogram)

        audio_file = spectrogram_to_audio(reverted_spectrogram)

        save_audio_to_folder(audio_file, sample_rate=22050, filename=('audio_file' + str(batch_idx) + '.wav'), folder='rock_generator/generated_audio')





if __name__ == "__main__":
    main()