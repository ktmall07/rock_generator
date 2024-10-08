import os

import torchaudio


def save_audio_to_folder(waveform, sample_rate, filename, folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Full path to save the audio file
    filepath = os.path.join(folder, filename)
    
    # Save the waveform as a WAV file
    torchaudio.save(filepath, waveform, sample_rate)
    print(f"Audio saved at: {filepath}")

    return