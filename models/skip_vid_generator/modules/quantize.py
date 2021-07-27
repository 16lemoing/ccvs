# Adapted from https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py

import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding per position
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    - mult : number of cat embeddings per position
    """

    def __init__(self, n_e, e_dim, beta, mult=1, normalize=False):
        super().__init__()
        self.n_e = n_e
        assert e_dim % mult == 0
        self.e_dim = e_dim // mult
        self.beta = beta
        self.mult = mult
        self.normalize = normalize

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if self.e_dim <= 1:
            self.embedding.weight.data.uniform_(0, 1.0)
        else:
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = [b, (t,) c, (h, w)]
        """
        # reshape z -> [b, (t, h, w), c] and flatten -> [b*(t*h*w), c]
        if z.ndim >= 4:
            z = z.transpose(-3, -1).transpose(-3, -2).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e, dtype=z.dtype).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        if self.normalize:
            z_q = z_q / torch.norm(z_q, p=2, dim=-1, keepdim=True)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        if z.ndim >= 4:
            z_q = z_q.transpose(-3, -1).transpose(-2, -1).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def embed_code(self, code):
        z = self.embedding(code)
        if self.mult > 1:
            s = list(z.shape)
            s[-1] *= self.mult
            s[-2] //= self.mult
            return z.view(s)
        return z