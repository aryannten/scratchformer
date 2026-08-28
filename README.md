# Scratchformer

A character-level Generative Pre-trained Transformer (GPT) language model implemented entirely from scratch in PyTorch, built as a transparent, educational reference for understanding the inner workings of transformer architectures.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Motivation](#project-motivation)
3. [Architecture](#architecture)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Usage](#usage)
   - [Data Preparation](#data-preparation)
   - [Training](#training)
   - [Text Generation](#text-generation)
   - [Interactive Web Demo](#interactive-web-demo)
8. [Configuration](#configuration)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)

---

## Overview

Scratchformer is a decoder-only transformer language model that operates at the character level. Every component — the tokenizer, multi-head causal self-attention, transformer blocks, training loop, and sampling strategies — is implemented from raw PyTorch tensor operations without relying on high-level libraries such as Hugging Face Transformers.

The default training corpus is a custom-built dataset of FIFA World Cup history, including tournament summaries, historic match narratives, legendary player biographies, and stadium information assembled from public Wikipedia extracts and historical CSV data sources.

**Default model specifications:**

| Hyperparameter | Value |
|----------------|-------|
| Tokenizer | Character-level |
| Vocabulary size | Approximately 65 to 100 unique characters |
| Embedding dimension | 128 |
| Number of layers | 4 |
| Number of attention heads | 4 |
| Context length | 128 tokens |
| Total parameters | Approximately 0.8 to 3 million |
| Optimizer | AdamW |
| Learning rate schedule | Linear warmup, cosine decay |

---

## Project Motivation

Modern large language models are typically accessed through abstracted APIs. This project was developed to demystify what happens beneath those abstractions. By constructing each component by hand — from the math behind causal masking to the logic of the autoregressive forward pass — Scratchformer serves as a concrete and transparent reference for understanding how transformers process sequential data and generate coherent text.

The codebase is intended for learners, educators, and practitioners who want to:

- Understand how the attention mechanism operates at the tensor level.
- Observe how gradient flow, residual connections, and layer normalization interact.
- See the complete training and inference loop without hidden abstractions.
- Experiment with custom datasets and hyperparameters in a small, readable codebase.

---

## Architecture

Scratchformer follows a standard decoder-only transformer architecture. Data flows through the network as follows:

```text
Input Text  (Characters)
     │
     ▼
[ Tokenizer ]  ->  Integer Token IDs
     │
     ▼
[ Token Embedding ] + [ Position Embedding ]  (Size: n_embd)
     │
     ▼
┌─────────────────────────────────────────┐
│ Transformer Block (Repeated N Times)    │
│   │                                     │
│   ├─> [ Layer Normalization ]           │
│   ├─> [ Multi-Head Causal Attention ]   │
│   ├─> [ Residual Add ]                  │
│   │                                     │
│   ├─> [ Layer Normalization ]           │
│   ├─> [ Feed-Forward Network ]          │
│   └─> [ Residual Add ]                  │
└─────────────────────────────────────────┘
     │
     ▼
[ Final Layer Normalization ]
     │
     ▼
[ Linear Projection ]  ->  Logits (Size: vocab_size)
     │
     ▼
[ Softmax ]  ->  Next Character Probabilities
```

**Key structural components:**

- **Custom Tokenizer**: A character-level tokenizer that maps raw text to integer token IDs and back, with serialization support for saving and loading vocabularies.
- **Embeddings**: Learned token and positional embeddings provide both semantic content and a sense of sequence order. Positional embeddings are essential because self-attention is permutation-invariant.
- **Self-Attention Mechanism**: Scaled dot-product attention with causal masking prevents future token visibility during next-character prediction.
- **Multi-Head Attention**: Parallel attention heads allow the model to focus on different linguistic patterns simultaneously. Outputs are concatenated and projected back to the embedding dimension.
- **Transformer Blocks**: Sequential blocks featuring pre-normalization, multi-head attention, feed-forward networks, and residual connections.
- **Feed-Forward Network**: A position-wise multi-layer perceptron expanding the embedding dimension by a factor of four, applying GELU activation, and projecting back.
- **Regularization**: Configurable dropout across embeddings, attention, and feed-forward layers to reduce overfitting.
- **Autoregressive Generation**: Supports greedy decoding, temperature scaling, and top-k sampling.

---

## Repository Structure

```
scratchformer/
|-- attention.py              # Single and multi-head self-attention implementation
|-- block.py                  # Transformer block (attention, FFN, LayerNorm, residuals)
|-- model.py                  # Full Scratchformer model assembly
|-- tokenizer.py              # Character-level tokenizer
|-- train.py                  # Training script with CLI
|-- generate.py               # Standalone text generation script
|-- prepare_data.py           # Dataset preparation and tokenization
|-- fetch_custom_data.py      # FIFA World Cup dataset builder
|-- demo_app.py               # Gradio interactive web interface
|-- train.ipynb               # Training notebook (Colab-compatible)
|-- train_custom.ipynb        # Training notebook for custom FIFA dataset
|-- requirements.txt          # Python dependencies
|-- scratchformer_plan.md     # Original 10-day build plan
|-- walkthrough.md            # Detailed day-by-day walkthrough
|-- data/
|   |-- raw/                  # Raw text corpora
|   `-- prepared/             # Tokenized train/val splits
|-- checkpoints/              # Saved model checkpoints
`-- tests/                    # Unit and integration tests
```

---

## Requirements

Python 3.8 or higher is required. All Python dependencies are listed in `requirements.txt`.

| Package    | Minimum Version | Purpose                          |
|------------|-----------------|----------------------------------|
| PyTorch    | 2.0             | Tensor operations and autograd   |
| NumPy      | any             | Numerical utilities              |
| Matplotlib | any             | Loss curve plotting              |
| tqdm       | any             | Training progress bars           |
| Gradio     | any             | Interactive web demo             |
| Requests   | any             | Dataset downloading              |

For GPU acceleration during training, a CUDA-compatible GPU with appropriate drivers and PyTorch CUDA support is recommended but not required. The model trains successfully on CPU for small configurations.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aryannten/scratchformer.git
cd scratchformer
```

### 2. Create and Activate a Virtual Environment

A virtual environment is strongly recommended to avoid conflicts with system-wide packages.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux and macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

Run the test suite to confirm everything is correctly installed:

```bash
python -m pytest tests/ -v
```

---

## Usage

The following sections describe the four primary workflows: data preparation, training, generation, and the interactive demo.

### Data Preparation

Scratchformer supports two built-in datasets: Tiny Shakespeare (a standard sanity-check corpus) and a custom FIFA World Cup corpus.

**Option A: Tiny Shakespeare (recommended for first-time users)**

```bash
python prepare_data.py --dataset shakespeare
```

This command downloads the Tiny Shakespeare corpus into `data/raw/`, tokenizes the text, splits it into 90 percent training and 10 percent validation, and saves the resulting tensors and vocabulary to `data/prepared/`.

**Option B: Custom FIFA World Cup Corpus**

First, build the custom corpus by fetching Wikipedia articles and historical match data:

```bash
python fetch_custom_data.py
```

This downloads approximately 60 Wikipedia articles covering all World Cup tournaments from 1930 to 2026, historic finals, legendary players, and tactical analyses. It also synthesizes match data into narrative reports. The resulting corpus is approximately 2.4 million characters of natural English football prose and is saved to `data/raw/custom_corpus.txt`.

Then tokenize and prepare the data:

```bash
python prepare_data.py --dataset custom
```

**Output artifacts:**

| File                                    | Description                          |
|-----------------------------------------|--------------------------------------|
| `data/prepared/train.pt`                | Training tensor (Shakespeare)        |
| `data/prepared/val.pt`                  | Validation tensor (Shakespeare)      |
| `data/prepared/vocab.json`              | Vocabulary mapping (Shakespeare)     |
| `data/prepared/custom_train.pt`         | Training tensor (custom)             |
| `data/prepared/custom_val.pt`           | Validation tensor (custom)           |
| `data/prepared/custom_vocab.json`       | Vocabulary mapping (custom)          |

### Training

The `train.py` script initializes the model and executes the training loop using the AdamW optimizer with a cosine learning rate decay schedule and linear warmup.

**Train on the custom FIFA dataset (recommended):**

```bash
python train.py --dataset custom --max-steps 5000 --batch-size 64 --dropout 0.1
```

**Train on Tiny Shakespeare:**

```bash
python train.py --dataset shakespeare --max-steps 5000 --batch-size 64
```

**Resume training from a checkpoint:**

```bash
python train.py --dataset custom --resume checkpoints/best.pt
```

**Outputs produced by training:**

- `checkpoints/best.pt` — Model with the lowest validation loss observed during training.
- `checkpoints/final.pt` — Model at the end of training.
- `checkpoints/step_<N>.pt` — Periodic checkpoints saved every `save_interval` steps (default 500). These are useful safety nets for long training runs on ephemeral compute environments.
- `checkpoints/loss_curve.png` — A plot of training and validation loss over time.

The loss curve is the most important diagnostic during training. If training loss decreases but validation loss increases, the model is overfitting and regularization should be increased. If both losses plateau early, the model may be too small or the learning rate too low. If loss is highly variable, the learning rate may be too high or the batch size too small.

### Text Generation

The `generate.py` script loads a trained checkpoint and produces text using a variety of sampling strategies.

**Standard generation with a prompt:**

```bash
python generate.py --prompt "The 1970 FIFA World Cup" --temperature 0.8 --max-tokens 250
```

**Greedy decoding (deterministic, most repetitive):**

```bash
python generate.py --greedy --prompt "Diego Maradona" --max-tokens 200
```

**Interactive REPL mode:**

```bash
python generate.py --interactive
```

In interactive mode, type a prompt and press Enter to generate text. The following commands are available:

| Command      | Description                              |
|--------------|------------------------------------------|
| `/temp X`    | Set sampling temperature to X            |
| `/topk X`    | Set top-k value to X                     |
| `/greedy`    | Toggle greedy decoding on or off         |
| `/tokens X`  | Set maximum number of generated tokens   |
| `/settings`  | Display current generation settings      |
| `/quit`      | Exit interactive mode                    |

**Temperature comparison mode:**

```bash
python generate.py --compare --prompt "Pele is widely considered" --max-tokens 200
```

This produces the same prompt at five different temperature settings (greedy, 0.3, 0.8, 1.0, and 1.5) for direct comparison of how temperature affects output quality.

**Multiple samples:**

```bash
python generate.py --prompt "In the final match" --num-samples 3 --max-tokens 150
```

**Using a specific checkpoint:**

```bash
python generate.py --checkpoint checkpoints/best.pt --prompt "Total Football" --max-tokens 200
```

**For reproducible generation:**

```bash
python generate.py --prompt "World Cup 1986" --seed 42 --max-tokens 200
```

### Interactive Web Demo

For an accessible graphical interface, a Gradio web application is provided.

```bash
python demo_app.py
```

This launches a local web server (typically at `http://127.0.0.1:7860`). Open this address in a web browser to access the interface, which provides:

- A text input for the prompt.
- A slider for **Max Tokens** (50 to 1000).
- A slider for **Temperature** (0.1 to 1.5), controlling creativity versus coherence.
- A slider for **Top-K** (1 to 100), restricting sampling to the most probable tokens.
- A set of example prompts demonstrating common use cases.

The demo expects `checkpoints/best.pt` and a vocabulary file in either `checkpoints/` or `data/prepared/`. If training was completed successfully, these files will already be in place.

---

## Configuration

All hyperparameters are exposed as dataclass fields for easy modification.

### Model Hyperparameters (`GPTConfig` in `model.py`)

| Field        | Default | Description                                |
|--------------|---------|--------------------------------------------|
| `vocab_size` | 65      | Number of unique tokens in the vocabulary  |
| `block_size` | 128     | Maximum context length in tokens           |
| `n_layer`    | 4       | Number of stacked transformer blocks       |
| `n_head`     | 4       | Number of attention heads per block         |
| `n_embd`     | 128     | Embedding dimension                        |
| `dropout`    | 0.0     | Dropout probability                        |

### Training Hyperparameters (`TrainConfig` in `train.py`)

| Field            | Default        | Description                                   |
|------------------|----------------|-----------------------------------------------|
| `dataset`        | shakespeare    | Dataset identifier                            |
| `max_steps`      | 5000           | Total number of training steps                |
| `batch_size`     | 64             | Number of sequences per batch                 |
| `learning_rate`  | 3e-4           | Peak learning rate for AdamW                  |
| `weight_decay`   | 0.1            | AdamW weight decay (regularization)           |
| `grad_clip`      | 1.0            | Maximum gradient norm for clipping            |
| `warmup_steps`   | 200            | Linear warmup duration in steps               |
| `min_lr`         | 3e-5           | Minimum learning rate after cosine decay      |
| `eval_interval`  | 250            | Steps between validation evaluations          |
| `eval_iters`     | 50             | Batches averaged per evaluation               |
| `save_interval`  | 500            | Steps between periodic checkpoints            |

### Command-Line Arguments for `train.py`

| Argument        | Default     | Description                                |
|-----------------|-------------|--------------------------------------------|
| `--dataset`     | shakespeare | One of shakespeare or custom               |
| `--max-steps`   | 5000        | Total training steps                       |
| `--batch-size`  | 64          | Sequences per batch                        |
| `--lr`          | 3e-4        | Peak learning rate                         |
| `--n-layer`     | 4           | Number of transformer blocks               |
| `--n-head`      | 4           | Attention heads per block                  |
| `--n-embd`      | 128         | Embedding dimension                        |
| `--block-size`  | 128         | Context window size                        |
| `--dropout`     | 0.0         | Dropout rate                               |
| `--resume`      | None        | Path to checkpoint to resume from          |

---

## Testing

The project includes a comprehensive test suite covering all major components.

```bash
python -m pytest tests/ -v
```

**Test coverage includes:**

- `tests/test_tokenizer.py` — Encode and decode roundtrip, vocabulary save and load.
- `tests/test_attention.py` — Single attention head shape verification, causal mask correctness.
- `tests/test_block.py` — FeedForward and TransformerBlock forward pass correctness.
- `tests/test_model.py` — Full model output shape, loss sanity check, parameter counting, embedding dimensions, gradient flow, generation correctness, temperature effects, context cropping.
- `tests/test_generate.py` — Generation output type, prompt preservation, token count accuracy, greedy determinism, seed reproducibility, integration test with a real checkpoint.

---

## Troubleshooting

**`RuntimeError: Training data not found at data/prepared/train.pt`**

The tokenized tensors have not been generated yet. Run the appropriate data preparation command first:

```bash
python prepare_data.py --dataset shakespeare
```

**`FileNotFoundError: Checkpoint not found at checkpoints/best.pt`**

The model has not been trained yet, or training did not complete. Run `train.py` before attempting generation.

**Loss is not decreasing during training**

Check the following:

- The learning rate may be too high or too low. Try values in the range 1e-4 to 1e-3.
- The batch size may be too small. Larger batches typically produce more stable gradients.
- The dataset may be too small. Character-level models require a reasonable corpus to learn patterns.
- Gradient clipping should be enabled (default 1.0) to prevent exploding gradients.

**Generated text is repetitive or low quality**

- Lower the temperature (for example, 0.5 to 0.8) for more conservative outputs.
- Increase top-k to allow more diverse sampling, or decrease it for more focused sampling.
- Train for more steps. Small models often need 5,000 to 20,000 steps to produce coherent output.
- Use the custom FIFA dataset with dropout enabled, which produces more diverse outputs than Tiny Shakespeare at the same model size.

**`UnicodeEncodeError` on Windows when fetching custom data**

The `fetch_custom_data.py` script forces UTF-8 encoding on Windows to avoid this issue. If you encounter encoding errors, ensure you are running Python 3.7 or higher and that the script has not been modified.

**Out-of-memory errors during training**

Reduce the batch size:

```bash
python train.py --dataset custom --batch-size 16
```

Or reduce the context length and embedding dimension:

```bash
python train.py --dataset custom --block-size 64 --n-embd 96
```

---

## License

This project is open-source and intended for educational purposes.

---

## Acknowledgments

This project builds on the foundational work of:

- Andrej Karpathy's nanoGPT and his "Let's build GPT: from scratch, in code, spelled out" tutorial, which provided the conceptual basis for the architecture.
- The Tiny Shakespeare dataset, a standard benchmark for character-level language models.
- The worldcup database by jfjelstul, which supplied historical match, goal, and award data.
- Wikipedia, the source of tournament, player, and tactical articles used to build the custom corpus.
- PyTorch, the deep learning framework that makes the implementation possible.

---

## Related Documentation

- `walkthrough.md` — A day-by-day journal of the build process, including training results and design decisions.
- `scratchformer_plan.md` — The original 10-day build plan outlining the development roadmap.
