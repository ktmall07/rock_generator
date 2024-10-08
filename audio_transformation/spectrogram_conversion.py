import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
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


def compute_log_magnitude_spectrogram(waveform, n_fft=6 * 256, hop_length=256, win_length=1024):
    # Compute Short-Time Fourier Transform (STFT)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True)
    
    # Compute the magnitude spectrogram
    magnitude_spectrogram = stft.abs()
    
    # Add a small epsilon to avoid log(0) issues
    epsilon = 1e-10
    
    # Convert to log-magnitude spectrogram
    log_magnitude_spectrogram = torch.log(magnitude_spectrogram + epsilon)
    
    return log_magnitude_spectrogram

def load_waveform(file_path, sample_rate=22050):
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = T.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    # waveform = torch.tensor(waveform).unsqueeze(0)  # Convert to PyTorch tensor and add batch dimension
    return waveform

def pad_spectrogram(spectrogram, target_length):
    # Pad with zeros if spectrogram length is less than target
    if spectrogram.shape[-1] < target_length:
        pad_width = target_length - spectrogram.shape[-1]
        padded_spectrogram = F.pad(spectrogram, (0, pad_width, 0, 0), mode='constant', value=0)
        return padded_spectrogram
    else:
        # Optionally truncate if the spectrogram is longer than target
        return spectrogram[:, :, :target_length]

# Load multiple audio files and create a batch of log-magnitude spectrograms
def process_audio_batch(file_paths, sample_rate=22050, n_fft=1024, hop_length=512, win_length=1024):
    spectrogram_batch = []
    
    for file_path in file_paths:
        waveform = load_waveform(file_path, sample_rate=sample_rate)
        log_magnitude_spec = compute_log_magnitude_spectrogram(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        spectrogram_batch.append(log_magnitude_spec)

    # Determine the maximum length for padding
    max_length = max(spec.shape[-1] for spec in spectrogram_batch)
    
    # Pad each spectrogram to the maximum length
    padded_spectrogram_batch = [pad_spectrogram(spec, max_length) for spec in spectrogram_batch]
    padded_spectrogram_batch = torch.stack(padded_spectrogram_batch)
    
    return padded_spectrogram_batch

def create_dataloader(spectrogram_batch, batch_size=32, shuffle=True):
    dataset = TensorDataset(spectrogram_batch)  # Create a TensorDataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def revert_spectrogram(log_magnitude_spectrogram):
    return torch.exp(log_magnitude_spectrogram)

def spectrogram_to_audio(spectrogram, n_fft=1024, hop_length=512, win_length=1024, num_iters=32):
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=n_fft, 
                                                   hop_length=hop_length, 
                                                   win_length=win_length, 
                                                   power=1.0, 
                                                   n_iter=num_iters)
    
    return griffin_lim(spectrogram)
