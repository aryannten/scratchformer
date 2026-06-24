# 🧠 Scratchformer

A character-level GPT language model built **entirely from scratch** using raw PyTorch tensor operations. No `transformers` library, no shortcuts — just pure understanding of the transformer architecture.

---

## What is this?

Scratchformer is a from-scratch implementation of a decoder-only transformer (GPT-style) that learns to generate text one character at a time. Every component — self-attention, multi-head attention, feed-forward networks, layer normalization, and residual connections — is hand-written to deeply understand how modern language models work under the hood.

The model is first validated on the **Tiny Shakespeare** dataset (the standard nanoGPT sanity-check corpus), then trained on a custom **FIFA World Cup 2026** dataset.

---

## Architecture

```
Input Text
    │
    ▼
┌──────────────────────┐
│  Token Embedding     │  Maps each character to a learned vector
│  + Position Embedding│  Adds positional information
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Transformer Block   │ ×N (stacked)
│  ┌────────────────┐  │
│  │ LayerNorm      │  │
│  │ Multi-Head     │  │
│  │ Self-Attention │  │  Q, K, V projections → scaled dot-product → causal mask
│  │ + Residual     │  │
│  ├────────────────┤  │
│  │ LayerNorm      │  │
│  │ Feed-Forward   │  │  Linear → GELU → Linear
│  │ + Residual     │  │
│  └────────────────┘  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Final LayerNorm     │
│  Linear → Logits     │  Output probabilities over vocabulary
└──────────────────────┘
```

### The Math

**Self-Attention:**

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

- **Q** (Query), **K** (Key), **V** (Value) are learned linear projections of the input
- Division by **√d_k** prevents softmax from saturating with large dot products
- **Causal masking** sets future positions to `-inf` before softmax, ensuring a token can only attend to itself and previous tokens (critical for autoregressive generation)

**Multi-Head Attention** runs several attention heads in parallel, each learning different relationships, then concatenates and projects back.

---

## Model Specifications

| Hyperparameter | Value |
|---|---|
| Tokenizer | Character-level |
| Vocab size | ~65 (Shakespeare) |
| Embedding dim | 128 |
| Num layers | 4 |
| Num attention heads | 4 |
| Context length (block size) | 128 |
| Total params | ~1–3M |

---

## Project Structure

```
scratchformer/
├── data/
│   ├── raw/                  # Raw text files (tinyshakespeare.txt, custom corpus)
│   └── prepared/             # Tokenized .pt tensors, train/val splits, vocab.json
├── tokenizer.py              # Character-level tokenizer (encode/decode/save/load)
├── attention.py              # Single head + multi-head self-attention from scratch
├── block.py                  # Transformer block (attention + FFN + LayerNorm + residuals)
├── model.py                  # Full Scratchformer model class (coming soon)
├── generate.py               # Sampling strategies: greedy, temperature, top-k (coming soon)
├── train.ipynb               # Training notebook — runs on Colab T4 GPU (coming soon)
├── demo_app.py               # Gradio demo app (coming soon)
├── checkpoints/              # Saved model checkpoints (gitignored)
├── tests/                    # Unit tests for each component
├── prepare_data.py           # Dataset download, tokenization, and split script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Tech Stack

- **PyTorch** — Tensor ops and autograd (architecture is hand-written, backprop is automatic)
- **NumPy** — Non-gradient array work
- **Matplotlib** — Loss curve plotting
- **tqdm** — Training progress bars
- **Gradio** — Final interactive demo UI
- **Google Colab** — T4 GPU for training

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
git clone https://github.com/aryannten/scratchformer.git
cd scratchformer
pip install -r requirements.txt
```

### Prepare Data

Downloads Tiny Shakespeare (~1.1MB), builds character vocabulary, splits train/val, and saves `.pt` tensors:

```bash
python prepare_data.py
```

### Run Tests

```bash
python -m tests.test_tokenizer     # Verify tokenizer roundtrip
python -m tests.test_attention     # Verify attention shapes + causal masking
python -m tests.test_block         # Verify transformer block shape preservation
```

---

