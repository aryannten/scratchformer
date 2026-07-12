"""
Tests for generate.py — Day 6

Tests the standalone generation module's functions:
    1. Checkpoint loading & model reconstruction
    2. Text generation with different sampling strategies
    3. Prompt encoding and handling
    4. Greedy vs. temperature vs. top-k behavior
"""

import os
import sys
import torch
import pytest

# Add project root to path so we can import everything
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Scratchformer, GPTConfig
from tokenizer import CharTokenizer
from generate import generate_text, load_model_from_checkpoint


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def model_and_tokenizer():
    """Create a small model and tokenizer for testing (no checkpoint needed)."""
    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = Scratchformer(config)
    model.eval()

    # Build a simple tokenizer from Shakespeare-like characters
    chars = list("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    tokenizer = CharTokenizer(chars=chars)

    return model, tokenizer


# ── Test: generate_text returns a string ────────────────────────────

def test_generate_returns_string(model_and_tokenizer):
    """generate_text should return a non-empty string."""
    model, tokenizer = model_and_tokenizer
    text = generate_text(model, tokenizer, max_new_tokens=20, device="cpu")
    assert isinstance(text, str)
    assert len(text) > 0


# ── Test: prompt is included in output ──────────────────────────────

def test_prompt_preserved_in_output(model_and_tokenizer):
    """The generated text should start with the given prompt."""
    model, tokenizer = model_and_tokenizer
    prompt = "ROMEO:"
    text = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=20, device="cpu")
    assert text.startswith(prompt), f"Expected output to start with '{prompt}', got: '{text[:30]}...'"


# ── Test: max_new_tokens controls length ────────────────────────────

def test_generation_length(model_and_tokenizer):
    """Output token count should be prompt tokens + max_new_tokens."""
    model, tokenizer = model_and_tokenizer
    prompt = "AB"
    max_new = 50

    text = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_new, device="cpu")
    # The output includes the prompt tokens + generated tokens
    prompt_tokens = tokenizer.encode(prompt)
    total_expected = len(prompt_tokens) + max_new

    # Decode should produce exactly total_expected characters
    # (since it's char-level, 1 token = 1 character)
    assert len(text) == total_expected, f"Expected {total_expected} chars, got {len(text)}"


# ── Test: greedy mode is deterministic ──────────────────────────────

def test_greedy_deterministic(model_and_tokenizer):
    """Greedy decoding should produce the same output every time."""
    model, tokenizer = model_and_tokenizer

    results = []
    for _ in range(3):
        text = generate_text(
            model, tokenizer,
            prompt="A",
            max_new_tokens=30,
            greedy=True,
            device="cpu",
        )
        results.append(text)

    assert results[0] == results[1] == results[2], \
        f"Greedy outputs differ: {results}"


# ── Test: temperature affects diversity ─────────────────────────────

def test_temperature_affects_diversity(model_and_tokenizer):
    """
    Lower temperature should produce less diverse outputs than higher.
    We test this by generating many samples and checking unique count.
    """
    model, tokenizer = model_and_tokenizer

    def get_unique_outputs(temp, n=10):
        outputs = set()
        for _ in range(n):
            text = generate_text(
                model, tokenizer,
                prompt="A",
                max_new_tokens=20,
                temperature=temp,
                top_k=None,
                device="cpu",
            )
            outputs.add(text)
        return len(outputs)

    # Very low temperature should produce fewer unique outputs
    low_unique = get_unique_outputs(0.01, n=10)
    high_unique = get_unique_outputs(2.0, n=10)

    # With an untrained model the logit distribution is near-uniform, so even
    # low temps may produce some variety. The key invariant is that high temp
    # should produce at least as many unique outputs as low temp.
    assert high_unique >= low_unique, \
        f"High temp ({high_unique} unique) should be ≥ low temp ({low_unique} unique)"


# ── Test: empty prompt works ────────────────────────────────────────

def test_empty_prompt(model_and_tokenizer):
    """Generation should work with an empty prompt (uses token 0 seed)."""
    model, tokenizer = model_and_tokenizer
    text = generate_text(model, tokenizer, prompt="", max_new_tokens=20, device="cpu")
    assert isinstance(text, str)
    assert len(text) > 0


# ── Test: top_k limits sampling ─────────────────────────────────────

def test_top_k_limits_sampling(model_and_tokenizer):
    """top_k=1 should be equivalent to greedy (always pick the best token)."""
    model, tokenizer = model_and_tokenizer

    # top_k=1 means we only consider the single most likely token = greedy
    results = []
    for _ in range(3):
        text = generate_text(
            model, tokenizer,
            prompt="A",
            max_new_tokens=30,
            temperature=1.0,
            top_k=1,
            device="cpu",
        )
        results.append(text)

    assert results[0] == results[1] == results[2], \
        f"top_k=1 should be deterministic like greedy, got: {results}"


# ── Test: seed reproducibility ──────────────────────────────────────

def test_seed_reproducibility(model_and_tokenizer):
    """Same seed should produce same output."""
    model, tokenizer = model_and_tokenizer

    def gen_with_seed(seed):
        torch.manual_seed(seed)
        return generate_text(
            model, tokenizer,
            prompt="A",
            max_new_tokens=30,
            temperature=0.8,
            top_k=40,
            device="cpu",
        )

    out1 = gen_with_seed(42)
    out2 = gen_with_seed(42)
    assert out1 == out2, f"Same seed should give same output:\n  {out1}\n  {out2}"


# ── Test: checkpoint loading (integration) ──────────────────────────

def test_checkpoint_loading_with_real_checkpoint():
    """
    Integration test: load the actual trained checkpoint if it exists.
    Skip if no checkpoint is available (e.g., in CI without training).
    """
    ckpt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints", "best.pt"
    )

    if not os.path.exists(ckpt_path):
        pytest.skip("No trained checkpoint available — skipping integration test")

    model, tokenizer, meta = load_model_from_checkpoint(ckpt_path, device="cpu")

    # Model should be in eval mode
    assert not model.training, "Model should be in eval mode after loading"

    # Tokenizer should have been loaded
    assert tokenizer.vocab_size > 0

    # Meta should contain training info
    assert 'step' in meta
    assert 'model_config' in meta

    # Generate some text to make sure the loaded model works
    text = generate_text(model, tokenizer, prompt="ROMEO:", max_new_tokens=50, device="cpu")
    assert text.startswith("ROMEO:")
    assert len(text) > len("ROMEO:")
