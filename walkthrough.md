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

## ⏳ Next Up: Day 5
*Colab training notebook + first training run on Tiny Shakespeare*

---
