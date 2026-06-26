"""
Scratchformer — Training Script (Day 5)

This is the training loop for our from-scratch GPT model.
It handles:
    1. Data loading (grabbing random chunks from the tokenized tensor)
    2. Forward pass → loss → backward → optimizer step
    3. Periodic evaluation on train/val sets
    4. Checkpoint saving (to Google Drive on Colab, or local checkpoints/ dir)
    5. Loss curve plotting

Can be run standalone:
    python train.py

Or imported and called from train.ipynb on Colab:
    from train import train, TrainConfig
"""

import os
import time
import math
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so it works headless on Colab
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from tqdm import tqdm

from tokenizer import CharTokenizer
from model import Scratchformer, GPTConfig


# ── Training Configuration ──────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    All training hyperparameters in one place.

    These are separate from the model hyperparameters (GPTConfig) because
    they control HOW we train, not WHAT the model looks like.
    """
    # ── Data ───────────────────────────────────────────────────────
    dataset: str = "shakespeare"          # "shakespeare" or "custom"
    data_dir: str = "data/prepared"       # where train.pt and val.pt live

    # ── Training loop ──────────────────────────────────────────────
    max_steps: int = 5000                 # total training steps
    batch_size: int = 64                  # number of sequences per batch
    learning_rate: float = 3e-4           # AdamW learning rate (standard for small transformers)
    weight_decay: float = 0.1             # AdamW weight decay (regularization)
    grad_clip: float = 1.0               # gradient clipping max norm (prevents exploding gradients)

    # ── Learning rate schedule ─────────────────────────────────────
    warmup_steps: int = 200               # linear warmup from 0 to learning_rate
    min_lr: float = 3e-5                  # minimum LR at end of cosine decay

    # ── Evaluation ─────────────────────────────────────────────────
    eval_interval: int = 250              # evaluate every N steps
    eval_iters: int = 50                  # number of batches to average for eval loss

    # ── Checkpointing ──────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"   # where to save model checkpoints
    save_interval: int = 500              # save checkpoint every N steps

    # ── Logging ────────────────────────────────────────────────────
    log_interval: int = 50                # print loss every N steps


# ── Data Loading ────────────────────────────────────────────────────

def load_data(config: TrainConfig, block_size: int):
    """
    Load the pre-tokenized train/val tensors from disk.

    Returns:
        train_data: 1D tensor of all training token IDs
        val_data:   1D tensor of all validation token IDs

    These are just flat arrays of integers. We'll grab random chunks
    of length block_size from them during training.
    """
    if config.dataset == "shakespeare":
        train_path = os.path.join(config.data_dir, "train.pt")
        val_path = os.path.join(config.data_dir, "val.pt")
    else:
        train_path = os.path.join(config.data_dir, "custom_train.pt")
        val_path = os.path.join(config.data_dir, "custom_val.pt")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}.\n"
            f"Run 'python prepare_data.py --dataset {config.dataset}' first!"
        )

    train_data = torch.load(train_path, weights_only=True)
    val_data = torch.load(val_path, weights_only=True)

    print(f"Loaded {config.dataset} dataset:")
    print(f"  Train: {len(train_data):,} tokens")
    print(f"  Val:   {len(val_data):,} tokens")

    return train_data, val_data


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    """
    Grab a random batch of training examples from the data.

    How it works:
        1. Pick `batch_size` random starting positions in the data
        2. For each position, grab `block_size` consecutive tokens as input (x)
        3. The target (y) is the same window shifted by one token
           (we're predicting the NEXT character at each position)

    Example (block_size=4):
        data = [18, 47, 56, 57, 58, 1, 15, 47, 58, ...]
        If random start = 2:
            x = [56, 57, 58, 1]     ← input
            y = [57, 58, 1, 15]     ← target (shifted by 1)
        So: given [56], predict 57
            given [56, 57], predict 58
            given [56, 57, 58], predict 1
            given [56, 57, 58, 1], predict 15
    """
    # Random starting indices, making sure we don't run off the end
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack into (batch_size, block_size) tensors
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    # Move to the right device (CPU or GPU)
    x, y = x.to(device), y.to(device)
    return x, y


# ── Evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config: TrainConfig, device: str):
    """
    Estimate train and val loss by averaging over several batches.

    We use @torch.no_grad() because we don't need gradients during evaluation.
    This saves memory and is faster.

    Why average over multiple batches?
        A single batch loss is noisy — it depends on which random chunk
        of text you happened to grab. Averaging over eval_iters batches
        gives a much more stable estimate.
    """
    model.eval()  # switch to eval mode (disables dropout if we had any)
    losses = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        batch_losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = get_batch(data, config.batch_size, model.config.block_size, device)
            _, loss = model(x, y)
            batch_losses[k] = loss.item()
        losses[split_name] = batch_losses.mean().item()

    model.train()  # switch back to training mode
    return losses


# ── Learning Rate Schedule ──────────────────────────────────────────

def get_lr(step: int, config: TrainConfig):
    """
    Cosine learning rate schedule with linear warmup.

    Why not just use a constant learning rate?
        - Warmup: At the start, the model's parameters are random. Large
          gradient updates early on can cause instability. Warming up
          gradually increases the LR so early updates are small and stable.
        - Cosine decay: As training progresses and the model gets closer
          to a good solution, we want smaller updates to fine-tune.
          Cosine decay smoothly decreases LR from max to min.

    Schedule:
        Steps 0 → warmup_steps:        Linear increase from 0 to learning_rate
        Steps warmup_steps → max_steps: Cosine decrease from learning_rate to min_lr
    """
    # Phase 1: Linear warmup
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    # Phase 2: Cosine decay
    # Clamp step so we don't go below min_lr after max_steps
    if step >= config.max_steps:
        return config.min_lr

    # Cosine annealing formula
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # ranges from 1.0 → 0.0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# ── Checkpoint Management ───────────────────────────────────────────

def save_checkpoint(model, optimizer, step, train_config, losses, path):
    """
    Save a training checkpoint.

    We save everything needed to resume training:
        - model weights (the learned parameters)
        - optimizer state (momentum, adaptive learning rates for each param)
        - current step (so we know where we left off)
        - configs (so we can reconstruct the model)
        - losses (for the loss curve)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'model_config': model.config,
        'train_config': train_config,
        'losses': losses,
    }
    torch.save(checkpoint, path)
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None):
    """Load a checkpoint and restore model (and optionally optimizer) state."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('step', 0), checkpoint.get('losses', {})


# ── Loss Curve Plotting ─────────────────────────────────────────────

def plot_loss_curve(loss_log, save_path="loss_curve.png"):
    """
    Plot training and validation loss over time.

    This is one of the most important diagnostic tools during training:
        - If train loss goes down but val loss goes up → overfitting
        - If both plateau early → model is too small or LR is too low
        - If loss is very spiky → LR is too high or batch size too small
        - If train and val track each other closely → healthy training
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    steps = [entry['step'] for entry in loss_log]
    train_losses = [entry['train'] for entry in loss_log]
    val_losses = [entry['val'] for entry in loss_log]

    ax.plot(steps, train_losses, label='Train Loss', color='#4ECDC4', linewidth=2)
    ax.plot(steps, val_losses, label='Val Loss', color='#FF6B6B', linewidth=2)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Scratchformer Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate final values
    if len(steps) > 0:
        ax.annotate(f'{train_losses[-1]:.3f}',
                    xy=(steps[-1], train_losses[-1]),
                    fontsize=10, color='#4ECDC4', fontweight='bold')
        ax.annotate(f'{val_losses[-1]:.3f}',
                    xy=(steps[-1], val_losses[-1]),
                    fontsize=10, color='#FF6B6B', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Loss curve saved → {save_path}")


# ── Main Training Loop ──────────────────────────────────────────────

def train(
    model_config: GPTConfig = None,
    train_config: TrainConfig = None,
    device: str = None,
    resume_from: str = None,
):
    """
    The main training function. This is the heart of Day 5.

    Args:
        model_config:  GPTConfig for the model architecture.
        train_config:  TrainConfig for training hyperparameters.
        device:        'cuda' or 'cpu'. Auto-detected if None.
        resume_from:   Path to a checkpoint to resume training from.

    Returns:
        model:         The trained model.
        loss_log:      List of dicts with step/train/val losses.
    """
    # ── Defaults ───────────────────────────────────────────────────
    if model_config is None:
        model_config = GPTConfig()
    if train_config is None:
        train_config = TrainConfig()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("🧠 SCRATCHFORMER — Training")
    print("=" * 60)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print()

    # ── Load tokenizer to get vocab size ───────────────────────────
    if train_config.dataset == "shakespeare":
        vocab_path = os.path.join(train_config.data_dir, "vocab.json")
    else:
        vocab_path = os.path.join(train_config.data_dir, "custom_vocab.json")

    tokenizer = CharTokenizer.load(vocab_path)
    model_config.vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {tokenizer.vocab_size} characters")
    print(f"Model config: {model_config}")
    print()

    # ── Load data ──────────────────────────────────────────────────
    train_data, val_data = load_data(train_config, model_config.block_size)
    print()

    # ── Create model ───────────────────────────────────────────────
    model = Scratchformer(model_config).to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.2f}M)")
    print()

    # ── Optimizer ──────────────────────────────────────────────────
    # AdamW is the standard optimizer for transformers.
    # Key differences from plain Adam:
    #   - Weight decay is decoupled from the gradient update
    #   - This provides better regularization
    #   - Standard choice: lr=3e-4, weight_decay=0.1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # ── Resume from checkpoint if specified ────────────────────────
    start_step = 0
    loss_log = []

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        start_step, saved_losses = load_checkpoint(resume_from, model, optimizer)
        if saved_losses:
            loss_log = saved_losses if isinstance(saved_losses, list) else []
        print(f"  Resumed at step {start_step}")
        print()

    # ── Training loop ──────────────────────────────────────────────
    print(f"Training for {train_config.max_steps} steps...")
    print(f"  Batch size: {train_config.batch_size}")
    print(f"  Block size: {model_config.block_size}")
    print(f"  LR: {train_config.learning_rate} (warmup {train_config.warmup_steps} steps, cosine decay to {train_config.min_lr})")
    print(f"  Eval every {train_config.eval_interval} steps, save every {train_config.save_interval} steps")
    print("-" * 60)

    model.train()
    best_val_loss = float('inf')
    t0 = time.time()

    pbar = tqdm(range(start_step, train_config.max_steps), desc="Training", ncols=100)
    for step in pbar:

        # ── Learning rate schedule ─────────────────────────────────
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ── Evaluate periodically ─────────────────────────────────
        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:
            losses = estimate_loss(model, train_data, val_data, train_config, device)
            loss_log.append({'step': step, 'train': losses['train'], 'val': losses['val']})

            elapsed = time.time() - t0
            pbar.write(
                f"  Step {step:5d} | "
                f"Train loss: {losses['train']:.4f} | "
                f"Val loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Track best val loss for best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                best_path = os.path.join(train_config.checkpoint_dir, "best.pt")
                save_checkpoint(model, optimizer, step, train_config, loss_log, best_path)

        # ── Forward pass ───────────────────────────────────────────
        x, y = get_batch(train_data, train_config.batch_size, model_config.block_size, device)
        logits, loss = model(x, y)

        # ── Backward pass ──────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)  # clear old gradients
        loss.backward()                         # compute new gradients

        # Gradient clipping: cap the total gradient norm to prevent
        # exploding gradients (very common in transformers)
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        optimizer.step()                        # update parameters

        # ── Update progress bar ────────────────────────────────────
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.1e}")

        # ── Periodic checkpoint ────────────────────────────────────
        if (step + 1) % train_config.save_interval == 0:
            ckpt_path = os.path.join(train_config.checkpoint_dir, f"step_{step + 1}.pt")
            save_checkpoint(model, optimizer, step + 1, train_config, loss_log, ckpt_path)

    # ── Final eval & checkpoint ────────────────────────────────────
    final_losses = estimate_loss(model, train_data, val_data, train_config, device)
    loss_log.append({'step': train_config.max_steps, 'train': final_losses['train'], 'val': final_losses['val']})

    final_path = os.path.join(train_config.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, train_config.max_steps, train_config, loss_log, final_path)

    # ── Plot loss curve ────────────────────────────────────────────
    curve_path = os.path.join(train_config.checkpoint_dir, "loss_curve.png")
    plot_loss_curve(loss_log, save_path=curve_path)

    # ── Summary ────────────────────────────────────────────────────
    total_time = time.time() - t0
    print()
    print("=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time:    {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Final train:   {final_losses['train']:.4f}")
    print(f"  Final val:     {final_losses['val']:.4f}")
    print(f"  Best val:      {best_val_loss:.4f}")
    print(f"  Checkpoints:   {train_config.checkpoint_dir}/")
    print(f"  Loss curve:    {curve_path}")
    print()

    return model, loss_log, tokenizer


# ── CLI Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Scratchformer")
    parser.add_argument("--dataset", type=str, default="shakespeare",
                        choices=["shakespeare", "custom"],
                        help="Dataset to train on")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-layer", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Context window (block size)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    model_cfg = GPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )

    train_cfg = TrainConfig(
        dataset=args.dataset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    model, losses, tokenizer = train(
        model_config=model_cfg,
        train_config=train_cfg,
        resume_from=args.resume,
    )

    # Quick generation test with the trained model
    print("\n🔤 Quick generation test:")
    print("-" * 40)
    device = next(model.parameters()).device
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with token 0
    generated = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
    print(tokenizer.decode(generated[0].tolist()))
