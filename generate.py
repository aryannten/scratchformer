"""
Scratchformer — Text Generation Script (Day 6)

Standalone inference script that loads a trained checkpoint and generates text.

Supports three sampling strategies:
    1. Greedy:      Always pick the most likely next token (temperature → 0).
                    Most deterministic but can be repetitive.
    2. Temperature: Scale logits before softmax to control randomness.
                    Low (<1.0) = conservative, High (>1.0) = creative.
    3. Top-k:       Only consider the k most probable tokens before sampling.
                    Prevents the model from picking very unlikely characters.

Usage examples:
    # Generate with defaults (temperature=0.8, top_k=40)
    python generate.py

    # Greedy decoding (deterministic)
    python generate.py --greedy

    # Creative generation with a prompt
    python generate.py --prompt "ROMEO:" --temperature 1.2 --max-tokens 500

    # Use best checkpoint instead of final
    python generate.py --checkpoint checkpoints/best.pt

    # Interactive mode — keep generating with new prompts
    python generate.py --interactive
"""

import os
import sys
import argparse
import torch

from tokenizer import CharTokenizer
from model import Scratchformer, GPTConfig


# ── Checkpoint Loading ──────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """
    Load a trained Scratchformer from a checkpoint file.

    The checkpoint contains:
        - model_state_dict: learned weights
        - model_config: GPTConfig used during training
        - train_config: TrainConfig (has dataset info for finding the vocab)
        - step: training step when saved
        - losses: training history

    Returns:
        model:      The loaded Scratchformer in eval mode.
        tokenizer:  The CharTokenizer used during training.
        meta:       Dict with step, losses, and configs for display.
    """
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Make sure you've trained the model first (python train.py)")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # ── Reconstruct the model from saved config ────────────────────
    model_config = checkpoint['model_config']
    print(f"  Model config: {model_config}")

    model = Scratchformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # inference mode — no dropout, no gradient tracking

    param_count = model.count_parameters()
    print(f"  Parameters: {param_count:,} ({param_count / 1e6:.2f}M)")

    # ── Load the matching tokenizer ────────────────────────────────
    # Figure out which vocab file to use from the train config
    train_config = checkpoint.get('train_config', None)
    if train_config is not None:
        data_dir = getattr(train_config, 'data_dir', 'data/prepared')
        dataset = getattr(train_config, 'dataset', 'shakespeare')
    else:
        data_dir = 'data/prepared'
        dataset = 'shakespeare'

    if dataset == "shakespeare":
        vocab_path = os.path.join(data_dir, "vocab.json")
    else:
        vocab_path = os.path.join(data_dir, "custom_vocab.json")

    if not os.path.exists(vocab_path):
        print(f"❌ Vocab not found: {vocab_path}")
        sys.exit(1)

    tokenizer = CharTokenizer.load(vocab_path)
    print(f"  Tokenizer: {tokenizer.vocab_size} characters")

    # ── Gather metadata for display ────────────────────────────────
    step = checkpoint.get('step', '?')
    losses = checkpoint.get('losses', [])
    meta = {
        'step': step,
        'losses': losses,
        'model_config': model_config,
        'train_config': train_config,
    }

    # Show training loss if available
    if isinstance(losses, list) and len(losses) > 0:
        last = losses[-1]
        print(f"  Trained for {step} steps")
        print(f"  Final loss — train: {last.get('train', '?'):.4f}, val: {last.get('val', '?'):.4f}")

    print()
    return model, tokenizer, meta


# ── Text Generation ─────────────────────────────────────────────────

def generate_text(
    model: Scratchformer,
    tokenizer: CharTokenizer,
    prompt: str = "",
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: int = 40,
    greedy: bool = False,
    device: str = "cpu",
):
    """
    Generate text from the model.

    Args:
        model:          Trained Scratchformer in eval mode.
        tokenizer:      CharTokenizer for encoding/decoding.
        prompt:         Starting text. Empty string = start from scratch
                        (uses token 0 as the seed).
        max_new_tokens: Number of characters to generate.
        temperature:    Sampling temperature. Ignored if greedy=True.
        top_k:          Top-k filtering. Ignored if greedy=True.
        greedy:         If True, always pick the most likely token.
        device:         'cpu' or 'cuda'.

    Returns:
        The generated text as a string (including the prompt).
    """
    # ── Encode the prompt (or use a zero-token seed) ───────────────
    if prompt:
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) == 0:
            # Prompt had no recognizable characters — fall back to seed
            token_ids = [0]
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        # Start from token 0 (newline in Shakespeare vocab)
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    # ── Configure sampling strategy ────────────────────────────────
    # Greedy = temperature so low it's effectively argmax.
    # We set temperature to a tiny value rather than special-casing,
    # because the model's generate() method handles it cleanly.
    if greedy:
        gen_temp = 1e-8   # near-zero → softmax becomes argmax
        gen_top_k = 1     # only consider the single best token
    else:
        gen_temp = temperature
        gen_top_k = top_k

    # ── Generate ───────────────────────────────────────────────────
    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=gen_temp,
            top_k=gen_top_k,
        )

    # ── Decode back to text ────────────────────────────────────────
    generated_tokens = output[0].tolist()
    text = tokenizer.decode(generated_tokens)

    return text


# ── Display Helpers ─────────────────────────────────────────────────

def print_header():
    """Print a nice header for the generation output."""
    print("=" * 60)
    print("🧠 SCRATCHFORMER — Text Generation")
    print("=" * 60)
    print()


def print_generation(text: str, label: str = "", prompt: str = ""):
    """Pretty-print generated text with optional label and prompt highlighting."""
    if label:
        print(f"── {label} ──")

    if prompt:
        # Highlight where the prompt ends and generation begins
        prompt_end = len(prompt)
        if text[:prompt_end] == prompt:
            print(f"\033[1m{text[:prompt_end]}\033[0m{text[prompt_end:]}")
        else:
            print(text)
    else:
        print(text)
    print()


# ── Interactive Mode ────────────────────────────────────────────────

def interactive_mode(model, tokenizer, args, device):
    """
    Interactive REPL for generating text with different prompts.

    Type a prompt and press Enter to generate. Special commands:
        /quit       — exit interactive mode
        /temp X     — change temperature to X
        /topk X     — change top-k to X
        /greedy     — toggle greedy mode
        /tokens X   — change max tokens to X
        /settings   — show current settings
    """
    temperature = args.temperature
    top_k = args.top_k
    greedy = args.greedy
    max_tokens = args.max_tokens

    print("🔄 Interactive Mode")
    print("   Type a prompt and press Enter to generate.")
    print("   Commands: /quit /temp X /topk X /greedy /tokens X /settings")
    print("-" * 60)
    print()

    while True:
        try:
            user_input = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        # ── Handle commands ────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("👋 Goodbye!")
                break
            elif cmd == "/temp" and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"  ✅ Temperature set to {temperature}")
                except ValueError:
                    print("  ❌ Usage: /temp 0.8")
            elif cmd == "/topk" and len(parts) > 1:
                try:
                    top_k = int(parts[1])
                    print(f"  ✅ Top-k set to {top_k}")
                except ValueError:
                    print("  ❌ Usage: /topk 40")
            elif cmd == "/greedy":
                greedy = not greedy
                print(f"  ✅ Greedy mode: {'ON' if greedy else 'OFF'}")
            elif cmd == "/tokens" and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"  ✅ Max tokens set to {max_tokens}")
                except ValueError:
                    print("  ❌ Usage: /tokens 300")
            elif cmd == "/settings":
                mode = "greedy" if greedy else f"temp={temperature}, top_k={top_k}"
                print(f"  Temperature: {temperature}")
                print(f"  Top-k:       {top_k}")
                print(f"  Greedy:      {greedy}")
                print(f"  Max tokens:  {max_tokens}")
                print(f"  Mode:        {mode}")
            else:
                print(f"  ❌ Unknown command: {cmd}")
            print()
            continue

        # ── Generate text ──────────────────────────────────────────
        text = generate_text(
            model, tokenizer,
            prompt=user_input,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
            device=device,
        )
        print()
        print_generation(text, prompt=user_input)


# ── Comparison Mode ─────────────────────────────────────────────────

def comparison_mode(model, tokenizer, prompt, max_tokens, device):
    """
    Generate the same prompt at multiple temperatures and show side-by-side.
    Great for understanding how temperature affects output quality.
    """
    print("🌡️  Temperature Comparison")
    print(f"   Prompt: \"{prompt or '(empty)'}\"")
    print(f"   Max tokens: {max_tokens}")
    print()

    configs = [
        ("Greedy (deterministic)", {"greedy": True}),
        ("Temperature 0.3 (conservative)", {"temperature": 0.3, "top_k": 40}),
        ("Temperature 0.8 (balanced)", {"temperature": 0.8, "top_k": 40}),
        ("Temperature 1.0 (standard)", {"temperature": 1.0, "top_k": 40}),
        ("Temperature 1.5 (creative)", {"temperature": 1.5, "top_k": None}),
    ]

    for label, kwargs in configs:
        text = generate_text(
            model, tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            device=device,
            **kwargs,
        )
        print_generation(text, label=label, prompt=prompt)
        print("-" * 60)


# ── CLI Entry Point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained Scratchformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py                                    # Default generation
  python generate.py --greedy                           # Deterministic output
  python generate.py --prompt "ROMEO:" --temp 0.8       # Start from a prompt
  python generate.py --interactive                      # Interactive REPL
  python generate.py --compare                          # Compare temperatures
  python generate.py --checkpoint checkpoints/best.pt   # Use best checkpoint
        """,
    )

    # Model loading
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to the model checkpoint (default: checkpoints/best.pt)",
    )

    # Generation parameters
    parser.add_argument(
        "--prompt", type=str, default="",
        help="Starting text for generation (default: empty = start from scratch)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300,
        help="Number of tokens to generate (default: 300)",
    )
    parser.add_argument(
        "--temperature", "--temp", type=float, default=0.8,
        help="Sampling temperature: <1.0 = conservative, >1.0 = creative (default: 0.8)",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling: only consider k most likely tokens (default: 40)",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use greedy decoding (always pick most likely token)",
    )

    # Modes
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Enter interactive mode — generate repeatedly with new prompts",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate the same prompt at multiple temperatures for comparison",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cpu' or 'cuda' (default: auto-detect)",
    )

    # Sampling
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible generation",
    )

    # Number of samples
    parser.add_argument(
        "--num-samples", "-n", type=int, default=1,
        help="Number of samples to generate (default: 1)",
    )

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────
    print_header()

    # Device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print()

    # Seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device == 'cuda':
            torch.cuda.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")
        print()

    # ── Load model ─────────────────────────────────────────────────
    model, tokenizer, meta = load_model_from_checkpoint(args.checkpoint, device)

    # ── Choose mode ────────────────────────────────────────────────
    if args.interactive:
        interactive_mode(model, tokenizer, args, device)

    elif args.compare:
        comparison_mode(model, tokenizer, args.prompt, args.max_tokens, device)

    else:
        # ── Standard generation ────────────────────────────────────
        mode_desc = "greedy" if args.greedy else f"temp={args.temperature}, top_k={args.top_k}"
        print(f"Generating {args.max_tokens} tokens | mode: {mode_desc}")
        if args.prompt:
            print(f"Prompt: \"{args.prompt}\"")
        print("-" * 60)
        print()

        for i in range(args.num_samples):
            label = f"Sample {i + 1}/{args.num_samples}" if args.num_samples > 1 else ""

            text = generate_text(
                model, tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                greedy=args.greedy,
                device=device,
            )
            print_generation(text, label=label, prompt=args.prompt)

            if args.num_samples > 1 and i < args.num_samples - 1:
                print("-" * 60)


if __name__ == "__main__":
    main()
