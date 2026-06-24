import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, embed_dim: int, head_dim: int, block_size: int):
        super().__init__()
        # Key, Query, Value linear projections (without bias as standard in GPT models)
        self.key   = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        
        # Causal mask - lower triangular matrix to prevent attending to future tokens
        # registered as a buffer so it is moved to the correct device but not treated as a parameter
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        # Input shape: (B, T, C) - Batch, Time (seq_len), Channels (embed_dim)
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_dim)
        q = self.query(x)  # (B, T, head_dim)
        v = self.value(x)  # (B, T, head_dim)

        # Scaled dot-product attention
        # Scale by 1 / sqrt(head_dim) to avoid large values and softmax saturation
        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale  # (B, T, T)

        # Causal masking: fill upper triangle of the attention weights matrix with -inf
        # so that they map to 0 after softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # Weighted sum of values
        out = wei @ v  # (B, T, head_dim)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, embed_dim: int, num_heads: int, block_size: int):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([
            Head(embed_dim, head_dim, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)  # output projection

    def forward(self, x):
        # Concatenate outputs from all attention heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Apply output projection
        out = self.proj(out)
        return out

