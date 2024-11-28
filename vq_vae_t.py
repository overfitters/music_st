import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging

LOGGER = logging.getLogger("vq_vae_model")


class VQEmbedding(nn.Module):
    """
    Vector Quantization embedding layer.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 /
                                            num_embeddings, 1 / num_embeddings)


class StyleEncoder(nn.Module):
    """
    Custom Style Encoder to handle additional arguments.
    """

    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, style_input, style_length):
        # Currently not using style_length, but you can include it in your logic
        return self.encoder(style_input), {}


class Decoder(nn.Module):
    """
    Custom Decoder that combines content and style embeddings and ensures the output size matches the input size.
    """

    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4,
                               stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4,
                               stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2,
                               padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, content_quantized, style_encoded):
        # Concatenate content and style embeddings along the feature dimension
        combined = torch.cat((content_quantized, style_encoded), dim=1)

        # Decode
        decoded = self.decoder(combined)

        # Resize decoded to match content_input size
        target_length = content_quantized.shape[2]
        decoded = nn.functional.interpolate(
            decoded, size=target_length, mode="linear", align_corners=True)

        return decoded


class Model(nn.Module):
    """
    VQ-VAE Model.
    """

    def __init__(self, content_encoder, vq, style_encoder, decoder):
        super().__init__()
        self.content_encoder = content_encoder
        self.vq = vq
        self.style_encoder = style_encoder
        self.decoder = decoder

    def forward(self, content_input, style_input, content_length, style_length, return_losses=False):
        # Content encoding and quantization
        content_encoded = self.content_encoder(content_input)
        print(f"Content Encoded Shape: {content_encoded.shape}")
        content_quantized, vq_losses = self.vq(content_encoded)
        print(f"Content Quantized Shape: {content_quantized.shape}")

        # Style encoding
        style_encoded, style_losses = self.style_encoder(
            style_input, style_length)
        print(f"Style Encoded Shape: {style_encoded.shape}")

        # Decoding
        decoded = self.decoder(content_quantized, style_encoded)
        print(
            f"Decoded Shape: {decoded.shape}, Content Input Shape: {content_input.shape}")

        if not return_losses:
            return decoded

        # Calculate losses
        losses = {
            "reconstruction": ((decoded - content_input) ** 2).mean(),
            **vq_losses,
            **style_losses
        }
        return decoded, losses


class Experiment:
    """
    Experiment class to handle training and validation.
    """

    def __init__(self, logdir, sr=22050, device="cuda"):
        self.logdir = logdir
        self.sr = sr
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=logdir)
        self.model = None
        self.optimizer = None

    def setup_model(self, content_encoder, vq, style_encoder, decoder):
        """
        Initialize the VQ-VAE model components.
        """
        self.model = Model(content_encoder, vq,
                           style_encoder, decoder).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, train_loader, val_loader=None, epochs=10, val_period=1):
        if self.model is None:
            raise ValueError(
                "Model has not been initialized. Call `setup_model` first.")

        self.model.train()
        for epoch in range(epochs):
            print(f"Starting epoch {epoch + 1}")
            for i, ((content, content_lengths), (style, style_lengths)) in enumerate(train_loader):
                content, style = content.to(self.device), style.to(self.device)
                content_lengths, style_lengths = content_lengths.to(
                    self.device), style_lengths.to(self.device)

                decoded, losses = self.model(
                    content, style, content_lengths, style_lengths, return_losses=True)

                total_loss = losses["reconstruction"]
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                if i % 10 == 0:
                    print(f"Step {i}, Loss: {total_loss.item()}")
                    self.writer.add_scalar(
                        "Train/Loss", total_loss.item(), epoch * len(train_loader) + i)

            if val_loader and epoch % val_period == 0:
                self.validate(val_loader, epoch)

    def validate(self, val_loader, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, ((content, content_lengths), (style, style_lengths)) in enumerate(val_loader):
                content, style = content.to(self.device), style.to(self.device)
                content_lengths, style_lengths = content_lengths.to(
                    self.device), style_lengths.to(self.device)

                decoded, losses = self.model(
                    content, style, content_lengths, style_lengths, return_losses=True)
                total_loss += losses["reconstruction"].item()

            avg_loss = total_loss / len(val_loader)
            print(f"Validation Loss: {avg_loss}")
            self.writer.add_scalar("Validation/Loss", avg_loss, epoch)

    def save_model(self, epoch):
        model_path = os.path.join(self.logdir, f"model_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
