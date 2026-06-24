import torch
import torch.nn as nn
from attention import MultiHeadAttention

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity (GELU) and projection back."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    One block of a transformer.
    Consists of:
    - Pre-LayerNorm 1 -> Multi-Head Self-Attention -> Residual addition
    - Pre-LayerNorm 2 -> Feed-Forward Network -> Residual addition
    """
    def __init__(self, embed_dim: int, num_heads: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x):
        # Pre-LayerNorm is applied before self-attention and feedforward layers
        # with residual connections adding the original inputs
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
