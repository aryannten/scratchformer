# 🧠 Scratchformer Walkthrough

This document tracks the step-by-step progress of building the Scratchformer project. It will be updated as we complete each day's tasks.

> [!NOTE]
> Scratchformer is a character-level GPT language model built entirely from scratch using raw PyTorch tensor operations, aimed at deeply understanding the transformer architecture.

---

## ✅ Day 1: Repo Setup & Tokenizer

**Goal:** Establish the project foundation, prepare data, and build the tokenizer.

**Accomplishments:**
- Set up the project structure and initialized the repository.
- Created `tokenizer.py` to handle character-level encoding and decoding. The tokenizer is responsible for converting raw text into integer tokens and vice-versa.
- Wrote `prepare_data.py` to handle downloading the Tiny Shakespeare dataset, tokenizing it, and creating the training and validation splits.
- Built a custom data pipeline (`fetch_custom_data.py`) to gather FIFA World Cup data from historical CSVs and Wikipedia, generating natural-language sentences for a domain-specific dataset.
- Added comprehensive unit tests for the tokenizer (`tests/test_tokenizer.py`) to ensure a perfect roundtrip (encode -> decode -> original string).

---

## ✅ Day 2: Single Attention Head

**Goal:** Implement the core self-attention mechanism from scratch.

**Accomplishments:**
- Created `attention.py` and implemented the `Head` class.
- Implemented the scaled dot-product attention math: `Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`.
- Set up the Query (Q), Key (K), and Value (V) linear projections.
- **Critical Implementation:** Added **causal masking**. A lower-triangular mask ensures that the softmax operation assigns a weight of `-inf` (or 0 probability) to future tokens, forcing the model to only look at past tokens when predicting the next one.
- Added unit tests for the single attention head to verify tensor shapes and ensure the causal mask is behaving correctly (`tests/test_attention.py`).

---

## ✅ Day 3: Multi-Head Attention & Transformer Block

**Goal:** Scale the attention mechanism and build the complete Transformer Block.

**Accomplishments:**
- Extended `attention.py` by implementing the `MultiHeadAttention` class. This runs multiple single attention heads in parallel, concatenates their outputs, and applies a final linear projection.
- Created `block.py` to house the rest of the transformer block components:
  - Implemented the `FeedForward` network: a simple multi-layer perceptron (Linear → GELU → Linear) that expands the dimensionality by a factor of 4 before projecting it back down.
  - Assembled the `Block` class, which chains together Layer Normalization, Multi-Head Attention, and the Feed-Forward network.
  - Implemented **Residual Connections** (adding the input of a sublayer to its output) to help gradients flow during deep network training.
- Added unit tests for the FeedForward network and the full Transformer Block (`tests/test_block.py`) to verify that the forward pass works and tensor shapes remain consistent.

---

## ✅ Day 4: Full Model Assembly

**Goal:** Wire everything together into a complete, trainable GPT language model.

**Key Concepts Learned:**

### GPTConfig (dataclass)
We introduced a `GPTConfig` dataclass to group all hyperparameters in one place:
- `vocab_size` — how many unique characters exist (65 for Shakespeare)
- `block_size` — maximum context length (128 tokens)
- `n_layer` — how many transformer blocks to stack (4)
- `n_head` — attention heads per block (4)
- `n_embd` — embedding dimension (128)

### Token + Position Embeddings
- **Token Embedding** (`nn.Embedding`): A lookup table that maps each character ID to a learned 128-dimensional vector. Token 0 → vector[0], Token 1 → vector[1], etc.
- **Position Embedding** (`nn.Embedding`): A separate lookup table where each *position* (0, 1, 2, ..., 127) gets its own vector. Without this, the model can't distinguish "cat sat" from "sat cat" because self-attention is permutation-invariant.
- We simply **add** them together: `x = tok_emb + pos_emb`. Broadcasting handles the batch dimension.

### The Forward Pass
The full data flow:
1. Input token IDs `(B, T)` → Token Embedding + Position Embedding → `(B, T, n_embd)`
2. Pass through N stacked `TransformerBlock`s → `(B, T, n_embd)`
3. Final `LayerNorm` → `(B, T, n_embd)`
4. Linear projection (`lm_head`) → `(B, T, vocab_size)` — these are the **logits**

### Loss Computation
- When targets are provided, we compute **cross-entropy loss**.
- At random initialization, the model assigns roughly equal probability to all characters, so the expected loss is `ln(vocab_size)` ≈ `ln(65)` ≈ **4.17**. Our test confirmed the initial loss was right in this range. This is a great sanity check — if your initial loss is wildly different, something is wrong.
- PyTorch's `F.cross_entropy` expects shapes `(N, C)` and `(N,)`, so we reshape `(B, T, vocab_size)` → `(B*T, vocab_size)`.

### Text Generation (`generate` method)
We implemented autoregressive generation with:
- **Context cropping**: The model can only see `block_size` tokens at a time, so we always crop the input to the last `block_size` tokens.
- **Temperature**: Divides logits before softmax. Low temp (e.g. 0.1) → more deterministic. High temp (e.g. 2.0) → more creative/random.
- **Top-k sampling**: Only considers the top k most likely tokens, preventing the model from picking very unlikely characters.

### Test Results
All **14 tests** passed, covering:
- Config defaults and custom values
- Output shape `(B, T, vocab_size)` verification
- Loss sanity check (~4.17 at random init)
- Parameter count (~1-3M range)
- Token and position embedding dimensions
- Variable sequence lengths (1, 4, 16, 32, 64)
- Gradient flow through all parameters
- End-to-end causal masking (future tokens don't affect past predictions)
- Generation shape, token range, temperature effects, and context cropping

---

## ✅ Day 5: Training Infrastructure + First Run

**Goal:** Build the training loop and run the first training on Tiny Shakespeare.

**Key Concepts Learned:**

### TrainConfig (dataclass)
Mirrors `GPTConfig` but controls *how* we train rather than *what* the model looks like:
- `max_steps=5000` — total gradient updates
- `batch_size=64` — sequences per batch
- `learning_rate=3e-4` — standard AdamW LR for small transformers
- `weight_decay=0.1` — decoupled regularization
- `grad_clip=1.0` — prevents exploding gradients
- `warmup_steps=200` — linear LR warmup for stability
- `eval_interval=250` — evaluate every N steps

### Data Batching
Each training step grabs random chunks from the tokenized tensor:
- Pick `batch_size` random starting indices
- Extract `block_size` consecutive tokens as input (`x`)
- Target (`y`) is the same window **shifted by one** — the next character at each position
- This is how the model learns next-token prediction

### Learning Rate Schedule — Cosine with Warmup
- **Warmup phase** (steps 0→200): Linear ramp from 0 to `3e-4`. Early random parameters need gentle updates.
- **Cosine decay** (steps 200→5000): Smoothly decreases LR to `3e-5`. As the model approaches convergence, smaller updates fine-tune without overshooting.

### AdamW Optimizer
Standard choice for transformers. Key difference from plain Adam:
- Weight decay is *decoupled* from the gradient update (better regularization)
- Momentum + adaptive per-parameter learning rates handle different parameter scales

### Gradient Clipping
Caps the total gradient norm at 1.0. Without this, occasional large gradients (common in attention layers) can destabilize training by making huge parameter updates.

### Evaluation Strategy
- `estimate_loss()` averages loss over 50 random batches (not just one) for a stable estimate
- Tracks both train and val loss to detect overfitting
- Best val checkpoint is saved automatically

### Checkpoint System
Saves everything needed to resume training:
- Model weights + optimizer state + current step + configs + loss history
- Periodic saves every 500 steps (safety net for Colab disconnects)
- Best val loss checkpoint + final checkpoint

### Files Created
- `train.py` — standalone training script with CLI args, importable by notebook
- `train.ipynb` — Colab notebook with GPU setup, sanity checks, training, loss plotting, generation, and Drive backup

### Local Sanity Check
Ran 50 steps locally on CPU to verify the full pipeline:
- Loss dropped from **4.26 → 3.69** (correct — random init is ~ln(65)=4.17)
- Checkpoints saved and loaded successfully
- Loss curve PNG generated
- Generation produced output (gibberish at 50 steps is expected)

### Colab T4 GPU Training — Results
Trained for 5000 steps on a Tesla T4 GPU with batch_size=64, block_size=128:

| Metric | Value |
|--------|-------|
| Final train loss | **1.5278** |
| Final val loss | **1.7107** |
| Parameters | ~824K (0.82M) |
| GPU | Tesla T4 |

**What the model learned:**
- Real Shakespeare character names (CLARENCE, LEONTES, BENVOLIO, CLAUDIO, DUKE VINCENTIO, ISABEL)
- Dialogue format with character names followed by colons
- Mostly real English words and old-timey phrasing
- Line breaks and verse structure

**Sample generation (temperature=0.8):**
```
DUKE VINCENTIO:
Who, my lady! what is in this one a man's ne'ers bishe.
Would are not the fall'd cheeks is well our of
The practor h
```

**Temperature effects observed:**
- **0.3 (conservative):** More repetitive but structured — "The will the done are to his soul here"
- **0.8 (balanced):** Best quality — real words, character names, coherent phrases
- **1.2 (creative):** More garbled but inventive — "Desir, alk ragglievy and batmerh ramenions"

✅ **Pipeline verified end-to-end.** The model is clearly learning English, Shakespeare's writing style, and the dialogue format. Any issues from here are data-related, not code bugs.

---

## ⏳ Next Up: Day 6
*Write standalone `generate.py` with greedy/temperature/top-k sampling. Load the trained checkpoint locally and verify generation quality.*

---
