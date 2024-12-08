import torch
from torch import nn


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, use_codebook_loss=True, axis=-1, init_mode="fan_out"):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._use_codebook_loss = use_codebook_loss
        nn.init.kaiming_uniform_(self.embedding.weight, mode=init_mode)
        self._axis = axis

    def forward(self, inp):
        if self._axis != -1:
            inp = inp.transpose(self._axis, -1)

        distances = (
            torch.sum(inp**2, axis=-1, keepdim=True)
            - 2 * torch.matmul(inp, self.embedding.weight.T)
            + torch.sum(self.embedding.weight**2, axis=-1)
        )
        ids = torch.argmin(distances, axis=-1)
        quantized = self.embedding(ids)

        losses = {"commitment": ((quantized.detach() - inp) ** 2).mean(axis=-1)}
        if self._use_codebook_loss:
            losses["codebook"] = ((quantized - inp.detach()) ** 2).mean(axis=-1)

            # Straight-through gradient estimator as in the VQ-VAE paper
            # No gradient for the codebook
            quantized = (quantized - inp).detach() + inp
        else:
            # Modified straight-through gradient estimator
            # The gradient of the result gets copied to both inputs (quantized and non-quantized)
            quantized = inp + quantized - inp.detach()

        if self._axis != -1:
            quantized = quantized.transpose(self._axis, -1).contiguous()

        return quantized, ids, losses
