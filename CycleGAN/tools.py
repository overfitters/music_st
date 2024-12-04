import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import librosa

class MusicStyleDataset(Dataset):
    def __init__(self, style_audio_path, content_audio_path, sample_rate=22050, transform=None, use_q_transform=False):
        """
        Args:
            style_audio_path (str): Path to the directory containing style audio files.
            content_audio_path (str): Path to the directory containing content audio files.
            sample_rate (int): Sampling rate for the audio files. Defaults to 22050 Hz.
            transform (callable): Transform to apply to the audio (e.g., Mel-Spectrogram). Defaults to None.
            use_q_transform (bool): Whether to use the Q-transform instead of Mel-Spectrogram. Defaults to False.
        """
        self.style_audios = self._load_audio_files(style_audio_path)
        self.content_audios = self._load_audio_files(content_audio_path)
        self.sample_rate = sample_rate  # Default: 22050 Hz
        self.use_q_transform = use_q_transform
        self.transform = transform or self._default_transform()
    
    def _load_audio_files(self, path):
        audio_files = []
        return audio_files
    
    def _default_transform(self):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,  # Common sample rate
            n_mels=128,  # Number of mel bands
            n_fft=2048,  # FFT window size
            hop_length=512  # Hop length between frames
        )
    
    def _clip_audio(self, audio):
        max_length = 30 * self.sample_rate  # 30 seconds in samples
        return audio[:, :max_length]  # Assuming mono; adjust for stereo
    
    def _q_transform(self, audio):
        cqt = librosa.cqt(
            audio.numpy(),
            sr=self.sample_rate,
            hop_length=512,
            fmin=32.70,  # Lowest frequency: C1
            n_bins=84,   # 7 octaves * 12 bins per octave
            bins_per_octave=12
        )
        return np.abs(cqt)  # Use magnitude of CQT
    
    def __len__(self):
        return min(len(self.style_audios), len(self.content_audios))
    
    def __getitem__(self, idx):
        style_audio = self._clip_audio(self.style_audios[idx])
        content_audio = self._clip_audio(self.content_audios[idx])

        if self.use_q_transform:
            # Apply Q-transform
            style_spec = self._q_transform(style_audio)
            content_spec = self._q_transform(content_audio)
        else:
            # Apply Mel-Spectrogram
            style_spec = self.transform(style_audio)
            content_spec = self.transform(content_audio)

        return {
            'style': torch.tensor(style_spec, dtype=torch.float32),
            'content': torch.tensor(content_spec, dtype=torch.float32)
        }

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(Generator, self).__init__()
        
        # Initial convolution layers
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_channels = 64
        for _ in range(2):
            out_channels = in_channels * 2
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        # Residual blocks
        for _ in range(9):
            model += [ResidualBlock(in_channels)]
        
        # Upsampling
        for _ in range(2):
            out_channels = in_channels // 2
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
        
        # Final layers
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, output_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        
        model = [
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        in_channels = 64
        for _ in range(3):
            out_channels = min(in_channels * 2, 512)
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_channels = out_channels
        
        model += [
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
