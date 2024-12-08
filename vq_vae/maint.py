import os
import random
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from vq_vae_model import Experiment, VQEmbedding, StyleEncoder, Decoder

# Step 1: Define FMA Dataset and Transformations


class FMAAudioDataset(Dataset):
    def __init__(self, audio_dir, sr=22050, segment_length=8, transform=None):
        """
        Dataset for FMA_small audio files.

        :param audio_dir: Directory containing FMA_small audio files.
        :param sr: Sampling rate for audio processing.
        :param segment_length: Length of segments to extract (in seconds).
        :param transform: Data augmentation transformations.
        """
        self.audio_dir = audio_dir
        self.sr = sr
        self.segment_length = segment_length
        self.transform = transform
        self.files = [
            os.path.join(root, file)
            for root, _, files in os.walk(audio_dir)
            for file in files if file.endswith('.mp3')
        ]
        self.files = [f for f in self.files if self._is_valid_audio(f)]

        if len(self.files) == 0:
            raise ValueError(
                f"No valid .mp3 files found in directory {audio_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        audio, _ = librosa.load(file_path, sr=self.sr)
        total_length = self.segment_length * self.sr

        # Randomly sample two non-overlapping segments
        start1 = random.randint(0, len(audio) - 2 * total_length)
        start2 = start1 + total_length + \
            random.randint(0, len(audio) - start1 - total_length)
        segment1 = audio[start1:start1 + total_length]
        segment2 = audio[start2:start2 + total_length]

        if self.transform:
            segment1, segment2 = self.transform(
                segment1), self.transform(segment2)

        # Reshape to add channel dimension
        segment1 = segment1[None, :]
        segment2 = segment2[None, :]

        return segment1, segment2

    def _is_valid_audio(self, file_path):
        try:
            audio, _ = librosa.load(file_path, sr=self.sr)
            return len(audio) >= 2 * self.segment_length * self.sr
        except Exception:
            return False


def audio_transform(audio):
    """
    Apply random timbre-altering transformations to the audio.
    """
    # Apply random pitch shift
    audio = librosa.effects.pitch_shift(
        audio, sr=22050, n_steps=random.uniform(-5, 5))

    # Apply random effects (e.g., time stretching)
    effects = [
        lambda x: librosa.effects.time_stretch(
            x, rate=random.uniform(0.8, 1.2)),  # Random stretch
        lambda x: x  # No effect
    ]
    effect = random.choice(effects)
    audio = effect(audio)
    return audio


def collate_padded_tuples(batch):
    """
    Collate function for padding variable-length audio segments.
    """
    segments1, segments2 = zip(*batch)
    max_length = max(max(seg.shape[1] for seg in segments1), max(
        seg.shape[1] for seg in segments2))
    padded_segments1 = torch.zeros(
        len(segments1), 1, max_length)  # Add channel dimension
    padded_segments2 = torch.zeros(
        len(segments2), 1, max_length)  # Add channel dimension
    for i, (seg1, seg2) in enumerate(zip(segments1, segments2)):
        padded_segments1[i, :, :seg1.shape[1]] = torch.tensor(seg1)
        padded_segments2[i, :, :seg2.shape[1]] = torch.tensor(seg2)
    return (padded_segments1, torch.tensor([seg1.shape[1] for seg1 in segments1])), \
           (padded_segments2, torch.tensor(
               [seg2.shape[1] for seg2 in segments2]))


def main():
    # Set paths
    audio_dir = "/content/fma_small/fma_small"  # Path to the FMA_small dataset
    logdir = "/content/logdir"  # Directory for logs and model outputs

    # Create dataset and dataloader
    fma_dataset = FMAAudioDataset(
        audio_dir=audio_dir, transform=audio_transform)
    fma_dataloader = DataLoader(
        fma_dataset, batch_size=16, shuffle=True, collate_fn=collate_padded_tuples)

    # Experiment setup
    exp = Experiment(
        logdir=logdir,
        sr=22050,
        device="cpu"  # Use GPU for training
    )

    # Define model components
    content_encoder = nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    )
    vq = VQEmbedding(num_embeddings=128, embedding_dim=128)
    style_encoder_layers = [
        nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
    ]
    style_encoder = StyleEncoder(style_encoder_layers)
    decoder = Decoder()  # Use updated Decoder class

    # Initialize the model
    exp.setup_model(content_encoder=content_encoder, vq=vq,
                    style_encoder=style_encoder, decoder=decoder)

    # Train the model
    print(f"Starting training with {len(fma_dataset)} audio files...")
    exp.train(train_loader=fma_dataloader, val_loader=fma_dataloader,
              epochs=10)  # Pass loaders directly


if __name__ == '__main__':
    main()
