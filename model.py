"""
Scratchformer — Full Model Assembly (Day 4)

This file is the culmination of Days 1-3. It takes every component we built
(tokenizer vocab size, attention heads, transformer blocks) and assembles them
into a complete, trainable GPT-style language model.

Architecture flow:
    Input token IDs  →  Token Embedding + Position Embedding
                     →  N × TransformerBlock
                     →  Final LayerNorm
                     →  Linear projection → Logits (vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from block import TransformerBlock


# ── Configuration ──────────────────────────────────────────────────

@dataclass
class GPTConfig:
    """
    All hyperparameters for the model in one place.

    Why a dataclass? It's cleaner than passing 6 separate arguments everywhere.
    You can create a config, print it, modify it, and pass it as a single object.

    Fields:
        vocab_size:   Number of unique tokens (characters in our case).
        block_size:   Maximum context length — how many tokens the model can
                      "see" at once. Also called the sequence length or context window.
        n_layer:      Number of transformer blocks stacked on top of each other.
                      More layers = more capacity to learn complex patterns,
                      but also more parameters and slower training.
        n_head:       Number of attention heads per block. Each head learns to
                      focus on different types of relationships between tokens.
        n_embd:       Embedding dimension — the size of the vector that represents
                      each token internally. This is the "width" of the model.
    """
    vocab_size: int = 65       # Tiny Shakespeare has ~65 unique characters
    block_size: int = 128      # Context window: 128 characters at a time
    n_layer: int = 4           # 4 transformer blocks stacked
    n_head: int = 4            # 4 attention heads per block
    n_embd: int = 128          # Each token is represented as a 128-dim vector


# ── Full Model ─────────────────────────────────────────────────────

class Scratchformer(nn.Module):
    """
    The complete GPT-style language model.

    This class wires together:
        1. Token Embedding:     Converts each token ID into a learned vector.
        2. Position Embedding:  Gives the model a sense of ORDER. Without this,
                                the model can't distinguish "cat sat" from "sat cat"
                                because attention is permutation-invariant.
        3. Transformer Blocks:  N stacked blocks, each applying self-attention
                                and a feed-forward network with residual connections.
        4. Final LayerNorm:     Stabilizes the output of the last block.
        5. Language Model Head: A linear layer that projects the final hidden state
                                to logits over the vocabulary (one score per character).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Embedding layers ───────────────────────────────────────
        # Token embedding: vocab_size → n_embd
        # Each of the ~65 characters gets its own learned 128-dim vector.
        # Think of it as a lookup table: token ID 0 → vector[0], ID 1 → vector[1], etc.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embedding: block_size → n_embd
        # Each POSITION (0, 1, 2, ..., 127) gets its own learned 128-dim vector.
        # This is how the model knows that token at position 0 is the "first" token.
        # Without this, the model would treat "hello" and "olleh" identically
        # because self-attention alone is permutation-invariant.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # ── Transformer blocks ─────────────────────────────────────
        # Stack N transformer blocks sequentially.
        # Each block refines the representation by attending to context and
        # processing through a feed-forward network.
        # nn.Sequential lets us chain them so data flows: block0 → block1 → ... → blockN
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embed_dim=config.n_embd,
                num_heads=config.n_head,
                block_size=config.block_size
            )
            for _ in range(config.n_layer)
        ])

        # ── Final layer norm ───────────────────────────────────────
        # Applied after all transformer blocks, before the output projection.
        # This is standard in GPT-2 and later models (pre-norm architecture).
        # It normalizes the final hidden states so the output projection
        # receives well-scaled inputs.
        self.ln_final = nn.LayerNorm(config.n_embd)

        # ── Language model head (output projection) ────────────────
        # Projects from embedding dimension (128) to vocabulary size (~65).
        # The output is a vector of "logits" — raw (unnormalized) scores
        # for each character in the vocabulary.
        # To get probabilities, you'd apply softmax to these logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx:     (B, T) tensor of token IDs, where B=batch size, T=sequence length.
                     Example: [[18, 47, 56, 57, 1, 15, ...], [...]]
            targets: (B, T) tensor of target token IDs for computing loss.
                     These are the "right answers" — the next token at each position.
                     If None, we skip loss computation (used during generation).

        Returns:
            logits: (B, T, vocab_size) — predicted scores for each position.
            loss:   scalar cross-entropy loss, or None if targets not provided.
        """
        B, T = idx.shape
        device = idx.device

        # ── Step 1: Embeddings ─────────────────────────────────────
        # Token embedding: look up each token ID in the embedding table
        tok_emb = self.token_embedding(idx)                           # (B, T, n_embd)

        # Position embedding: create position indices [0, 1, 2, ..., T-1]
        # and look them up in the position embedding table.
        # torch.arange(T) creates [0, 1, 2, ..., T-1] on the correct device.
        pos_emb = self.position_embedding(torch.arange(T, device=device))  # (T, n_embd)

        # Add token + position embeddings together.
        # pos_emb is (T, n_embd) and tok_emb is (B, T, n_embd).
        # Broadcasting handles the batch dimension automatically:
        # each sample in the batch gets the same positional information added.
        x = tok_emb + pos_emb                                        # (B, T, n_embd)

        # ── Step 2: Transformer blocks ─────────────────────────────
        # Pass through all N stacked transformer blocks.
        # Each block applies: LayerNorm → MultiHeadAttention → Residual
        #                     LayerNorm → FeedForward → Residual
        x = self.blocks(x)                                           # (B, T, n_embd)

        # ── Step 3: Final LayerNorm ────────────────────────────────
        x = self.ln_final(x)                                         # (B, T, n_embd)

        # ── Step 4: Project to vocabulary ──────────────────────────
        # Each position now has a hidden state of size n_embd (128).
        # We project it to vocab_size (~65) to get one score per character.
        # The highest score = the model's best guess for the next character.
        logits = self.lm_head(x)                                     # (B, T, vocab_size)

        # ── Step 5: Compute loss (if targets provided) ─────────────
        if targets is not None:
            # Cross-entropy loss measures how far the model's predictions
            # are from the actual next characters.
            #
            # PyTorch's cross_entropy expects:
            #   input:  (N, C) where N = number of predictions, C = number of classes
            #   target: (N,) where each value is the correct class index
            #
            # We have logits as (B, T, vocab_size) and targets as (B, T),
            # so we reshape by collapsing B and T into one dimension.
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)      # (B*T, vocab_size)
            targets_flat = targets.view(B * T)        # (B*T,)
            loss = F.cross_entropy(logits_flat, targets_flat)
        else:
            loss = None

        return logits, loss

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        This is the inference loop. Starting from a prompt (idx), we repeatedly:
            1. Crop the context to the last block_size tokens (model can't see more)
            2. Run a forward pass to get logits for the next token
            3. Sample from the distribution (with temperature and optional top-k)
            4. Append the sampled token to the sequence
            5. Repeat

        Args:
            idx:             (B, T) starting token IDs (the prompt).
            max_new_tokens:  How many new tokens to generate.
            temperature:     Controls randomness. 1.0 = normal, <1.0 = more
                             deterministic (sharper distribution), >1.0 = more random.
            top_k:           If set, only sample from the top-k most likely tokens.
                             This prevents the model from picking very unlikely tokens.

        Returns:
            (B, T + max_new_tokens) tensor with the generated sequence.
        """
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens.
            # The model's positional embeddings only go up to block_size,
            # so we can never feed it more than that.
            idx_cond = idx[:, -self.config.block_size:]

            # Forward pass — we only need logits, not loss
            logits, _ = self(idx_cond)

            # Focus only on the LAST time step's logits.
            # We only care about what comes next after the last token.
            logits = logits[:, -1, :]                        # (B, vocab_size)

            # Apply temperature scaling
            # Dividing logits by temperature before softmax:
            #   - temperature < 1.0 → logits become larger → softmax is "peakier"
            #     → model picks the most likely token more often (more deterministic)
            #   - temperature > 1.0 → logits become smaller → softmax is "flatter"
            #     → model explores more unlikely tokens (more creative/random)
            if temperature != 1.0:
                logits = logits / temperature

            # Optional top-k filtering
            # Zero out all logits except the k highest ones.
            # This prevents the model from ever picking a very unlikely character.
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # v[:, [-1]] is the k-th largest value (the cutoff threshold)
                logits[logits < v[:, [-1]]] = float('-inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)                # (B, vocab_size)

            # Sample one token from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)          # (B, T+1)

        return idx
