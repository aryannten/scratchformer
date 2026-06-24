# Tiny GPT From Scratch — 10-Day Build Plan

**Goal:** Build a working transformer language model — tokenizer, embeddings, multi-head self-attention, feed-forward layers, layer norm, residual connections, and training loop — entirely from raw PyTorch tensor ops. No `transformers` library for the model itself. Train it on Colab's free T4 GPU, edit everything locally in Antigravity.

**De-risking strategy:** Verify your pipeline on a known-good dataset (tiny Shakespeare) before training on your own custom dataset (DSA explanations or F1 commentary). This way, if something looks broken later, you already know your code works and the problem is data-related, not a bug in your transformer implementation.

---

## Repo structure

```
scratchformer/
├── data/
│   ├── raw/                  # raw text files (tinyshakespeare.txt, your custom corpus)
│   └── prepared/             # tokenized .bin/.pt files, train/val splits
├── tokenizer.py               # char-level tokenizer (encode/decode)
├── attention.py                # single + multi-head self-attention from scratch
├── block.py                    # transformer block (attention + FFN + layernorm + residuals)
├── model.py                    # full Scratchformer model class
├── generate.py                  # sampling strategies (greedy, temperature, top-k)
├── train.ipynb                 # training notebook — runs on Colab GPU kernel
├── demo_app.py                  # Gradio app for the final demo
├── checkpoints/                 # local copies of downloaded checkpoints (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

**.gitignore** should include: `checkpoints/*.pt`, `data/raw/*.txt` (if large), `__pycache__/`, `.ipynb_checkpoints/`

---

## Tech stack

- **PyTorch** — tensor ops, autograd (you're not hand-rolling backprop, just the forward architecture)
- **NumPy** — for any non-gradient-tracked array work
- **Matplotlib** — plotting loss curves
- **tqdm** — training progress bars
- **Gradio** — final demo UI
- **Git** — bridges Antigravity (local) and Colab (remote) since the remote kernel can't see local files directly

---

## Antigravity + Colab workflow (recap)

1. Write/edit all `.py` files locally in Antigravity
2. `git push` after meaningful changes
3. In `train.ipynb` (kernel connected to Colab GPU), the first cell always does:

```python
!rm -rf scratchformer
!git clone https://github.com/yourusername/scratchformer.git
%cd scratchformer
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')
CHECKPOINT_DIR = '/content/drive/MyDrive/scratchformer_checkpoints'
```

4. Training loop saves checkpoints to `CHECKPOINT_DIR` every N steps — this is your bridge back to persistent storage since the Colab runtime is ephemeral
5. After training, download final checkpoint from Drive to your local `checkpoints/` folder for the demo app

---

## Model spec (keep it small — this is intentional)

| Hyperparameter | Value | Why |
|---|---|---|
| Tokenizer | Character-level | Simplest correct implementation, no BPE complexity to debug |
| Vocab size | ~65–100 | Depends on dataset's unique characters |
| Embedding dim | 128–192 | Small enough to train fast on T4 |
| Num layers | 4–6 | Enough depth to show real transformer behavior |
| Num attention heads | 4–6 | Standard ratio to embedding dim |
| Context length (block size) | 128–256 | Keep attention matrix manageable |
| Total params | ~1–10M | Trains in well under an hour per run on a T4 |

You can scale these up later if time allows — but get the small version fully working first.

---

## Day-by-day plan

### Day 1 — Repo setup + tokenizer
- Set up GitHub repo, clone into Antigravity workspace
- Download **tiny Shakespeare** dataset (the standard nanoGPT sanity-check corpus) into `data/raw/`
- Write `tokenizer.py`: character-level encode/decode functions, build vocab from unique characters
- Write train/val split logic, save prepared `.pt` tensors to `data/prepared/`
- **No GPU needed today** — all of this runs fine locally in Antigravity on CPU
- ✅ Checkpoint: you can encode a string, decode it back, and get the exact original text

### Day 2 — Single attention head from scratch
- Write `attention.py`: implement one self-attention head — Q, K, V linear projections, scaled dot-product attention, **causal masking** (critical: a token must not attend to future tokens)
- Test on a toy random tensor (e.g. batch=2, seq_len=8, embed_dim=32) — verify output shape matches input shape
- Write a sanity check: confirm masking works by checking attention weights for future positions are exactly zero after softmax
- **Still CPU-only, still local in Antigravity**
- ✅ Checkpoint: single attention head produces correctly-shaped output and respects causality

### Day 3 — Multi-head attention + feed-forward
- Extend `attention.py` to multi-head: run several attention heads in parallel, concatenate, project back down
- Write `block.py`: feed-forward (MLP) sublayer, LayerNorm, residual connections
- Assemble one full `TransformerBlock` class combining attention + FFN + norms + residuals
- Test forward pass shapes on toy input
- ✅ Checkpoint: one transformer block runs end-to-end on dummy data with correct output shape

### Day 4 — Full model assembly
- Write `model.py`: `Scratchformer` class — token embedding + positional embedding + N stacked transformer blocks + final layer norm + output projection to vocab logits
- Implement cross-entropy loss for next-token prediction
- Test forward pass + loss computation on a small real batch from your tiny Shakespeare data (still CPU, just verifying no shape bugs — don't worry about speed yet)
- ✅ Checkpoint: model produces logits of shape `(batch, seq_len, vocab_size)` and loss is a single sane number (~log(vocab_size) at random init, e.g. ~4.2 for 65 chars)

### Day 5 — Colab connection + first training run
- Install the Colab extension in Antigravity, connect `train.ipynb` to a T4 GPU runtime
- Write the training loop: forward pass → loss → backward → optimizer step (AdamW), with periodic checkpoint saves to Drive
- Push code, run the git-clone-on-Colab boilerplate, kick off your **first real training run** on tiny Shakespeare
- Let it run — go do something else, check back periodically
- ✅ Checkpoint: loss is visibly decreasing over steps (plot it)

### Day 6 — Debug, tune, verify generation
- Review loss curve — tune learning rate / batch size / gradient clipping if it's unstable or plateauing too early
- Write `generate.py`: greedy decoding, temperature sampling, top-k sampling
- Load your tiny-Shakespeare checkpoint and generate text — confirm it produces semi-coherent Shakespeare-style output (it won't be perfect, but it should look like English with old-timey phrasing, not random characters)
- **This is your big milestone**: pipeline is verified end-to-end. Any issues from here are about your custom data, not your code.
- ✅ Checkpoint: generated text is clearly more structured than random noise

### Day 7 — Swap in your custom dataset
- Build your real corpus: either
  - **DSA angle**: compile pattern explanations from your A2Z sheet notes (sliding window, two-pointer, DP explanations) into one text file, or
  - **F1 angle**: compile race regulations / strategy explanation text
- Rebuild vocab with `tokenizer.py` on the new corpus (vocab will differ from Shakespeare)
- Kick off a new training run on Colab with the custom dataset
- ✅ Checkpoint: training run started successfully, loss decreasing

### Day 8 — Iterate on custom training
- Monitor and tune the custom-data run — these usually need more iteration than the toy dataset since your corpus is smaller/more specialized
- If time allows, try a second variant (different hyperparams, or a slightly larger/cleaned dataset) for comparison
- Save your best final checkpoint
- ✅ Checkpoint: final checkpoint chosen, generation output reviewed and reads coherently for your domain

### Day 9 — Demo build
- Write `demo_app.py`: a Gradio interface — text box for a prompt, generates continuation using your trained model
- Test locally in Antigravity with your downloaded checkpoint
- Optional stretch: deploy to HuggingFace Spaces (free) for a shareable live link
- ✅ Checkpoint: working demo you can show someone in 30 seconds

### Day 10 — Polish + README + wrap-up
- Write `README.md`: explain the architecture, the math (attention formula, why causal masking matters), how training works, link to demo, sample outputs
- Record a short screen-capture demo video
- Final `git push`, clean up repo structure
- Update resume/portfolio with this project
- ✅ Checkpoint: repo is interview-ready — someone could read the README and understand exactly what you built and why it's impressive

---

## Things that commonly go wrong (so you're not surprised)

- **Shape mismatches in attention** — the most common bug. Print tensor shapes liberally on days 2-4.
- **Forgetting the causal mask** — if you skip it, the model "cheats" by looking at future tokens and your loss will look great but generation will be garbage.
- **Colab session timeouts** — save checkpoints frequently (every few hundred steps), not just at the end.
- **Custom dataset too small** — character-level models need a reasonable amount of text to learn patterns. If your DSA/F1 corpus is tiny, consider padding it with related text (multiple explanations, repeated phrasing) to give the model more to learn from.

---

## What "done" looks like

A GitHub repo with clean, readable from-scratch transformer code, a README explaining the architecture and math, a trained model that generates coherent domain-specific text, and a live demo link — something you can pull up on your phone and show someone in an interview within 30 seconds.
