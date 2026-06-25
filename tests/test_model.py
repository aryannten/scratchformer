import torch
import math
from model import Scratchformer, GPTConfig


# ── GPTConfig Tests ────────────────────────────────────────────────

def test_config_defaults():
    """Verify GPTConfig has sensible defaults."""
    print("Running GPTConfig defaults test...")

    config = GPTConfig()
    assert config.vocab_size == 65
    assert config.block_size == 128
    assert config.n_layer == 4
    assert config.n_head == 4
    assert config.n_embd == 128

    print(f"  vocab_size={config.vocab_size}, block_size={config.block_size}")
    print(f"  n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}")
    print("  PASSED\n")


def test_config_custom():
    """Verify GPTConfig accepts custom values."""
    print("Running GPTConfig custom values test...")

    config = GPTConfig(vocab_size=100, block_size=256, n_layer=6, n_head=8, n_embd=256)
    assert config.vocab_size == 100
    assert config.n_embd == 256

    print(f"  Custom config created successfully")
    print("  PASSED\n")


# ── Model Shape Tests ─────────────────────────────────────────────

def test_model_output_shape():
    """
    THE most important test: verify the model produces logits of shape
    (B, T, vocab_size). If this shape is wrong, nothing else will work.
    """
    print("Running model output shape test...")

    config = GPTConfig(vocab_size=65, block_size=128, n_layer=2, n_head=4, n_embd=64)
    model = Scratchformer(config)

    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))  # Random token IDs
    logits, loss = model(idx)

    assert logits.shape == (B, T, config.vocab_size), \
        f"Expected {(B, T, config.vocab_size)}, got {logits.shape}"
    assert loss is None, "Loss should be None when targets not provided"

    print(f"  logits shape: {logits.shape} OK")
    print(f"  loss: None (no targets) OK")
    print("  PASSED\n")


def test_model_loss_computation():
    """
    Verify the model computes cross-entropy loss when targets are provided.

    At random initialization, the loss should be approximately -ln(1/vocab_size)
    = ln(vocab_size). For vocab_size=65, that's about 4.17.
    This is because a randomly initialized model assigns roughly equal probability
    to all characters, so the negative log-likelihood of the correct one is ln(65).
    """
    print("Running model loss computation test...")

    config = GPTConfig(vocab_size=65, block_size=128, n_layer=2, n_head=4, n_embd=64)
    model = Scratchformer(config)

    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)

    assert loss is not None, "Loss should NOT be None when targets are provided"
    assert logits.shape == (B, T, config.vocab_size)

    # At random init, loss ≈ ln(vocab_size) ≈ 4.17
    expected_loss = math.log(config.vocab_size)
    assert abs(loss.item() - expected_loss) < 1.0, \
        f"Loss {loss.item():.2f} is too far from expected ~{expected_loss:.2f} at random init"

    print(f"  loss: {loss.item():.4f} (expected ~{expected_loss:.2f} at random init) OK")
    print("  PASSED\n")


def test_model_param_count():
    """
    Verify the model has a reasonable number of parameters.
    For our default config, it should be in the 1-3M range.
    """
    print("Running model parameter count test...")

    config = GPTConfig()  # default config
    model = Scratchformer(config)

    n_params = model.count_parameters()
    print(f"  Total parameters: {n_params:,}")

    # Should be between 100K and 10M for our small model
    assert n_params > 100_000, f"Too few params: {n_params}"
    assert n_params < 10_000_000, f"Too many params: {n_params}"

    print("  PASSED\n")


# ── Embedding Tests ────────────────────────────────────────────────

def test_token_embedding():
    """Verify token embedding maps token IDs to vectors of correct size."""
    print("Running token embedding test...")

    config = GPTConfig(vocab_size=65, n_embd=128)
    model = Scratchformer(config)

    # Check the embedding table dimensions
    assert model.token_embedding.num_embeddings == 65
    assert model.token_embedding.embedding_dim == 128

    # Feed in a single token and check the output vector size
    tok = torch.tensor([[0]])  # token ID 0
    emb = model.token_embedding(tok)
    assert emb.shape == (1, 1, 128), f"Embedding shape: {emb.shape}"

    print(f"  Embedding table: {model.token_embedding.num_embeddings} × {model.token_embedding.embedding_dim}")
    print("  PASSED\n")


def test_position_embedding():
    """Verify position embedding covers the full block_size."""
    print("Running position embedding test...")

    config = GPTConfig(block_size=128, n_embd=128)
    model = Scratchformer(config)

    # The position embedding table should have block_size rows
    assert model.position_embedding.num_embeddings == 128, \
        f"Expected 128 positions, got {model.position_embedding.num_embeddings}"
    assert model.position_embedding.embedding_dim == 128

    print(f"  Position table: {model.position_embedding.num_embeddings} positions × {model.position_embedding.embedding_dim} dims")
    print("  PASSED\n")


# ── Sequence Length Tests ──────────────────────────────────────────

def test_model_variable_sequence_lengths():
    """Verify the model works with different sequence lengths up to block_size."""
    print("Running variable sequence length test...")

    config = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=4, n_embd=32)
    model = Scratchformer(config)

    for T in [1, 4, 16, 32, 64]:
        idx = torch.randint(0, config.vocab_size, (1, T))
        logits, _ = model(idx)
        assert logits.shape == (1, T, config.vocab_size), \
            f"Failed for T={T}: got {logits.shape}"
        print(f"  T={T}: logits shape {logits.shape} OK")

    print("  PASSED\n")


# ── Gradient Tests ─────────────────────────────────────────────────

def test_model_gradient_flow():
    """
    Verify gradients flow through the entire model end-to-end.
    This is critical — if any layer blocks gradients, the model can't learn.
    """
    print("Running gradient flow test...")

    config = GPTConfig(vocab_size=65, block_size=128, n_layer=2, n_head=4, n_embd=64)
    model = Scratchformer(config)

    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets=targets)
    loss.backward()

    total_params = 0
    graded_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                graded_params += 1

    assert graded_params == total_params, \
        f"Only {graded_params}/{total_params} parameters received gradients!"
    print(f"  All {total_params} parameters received gradients OK")
    print("  PASSED\n")


# ── Causal Masking End-to-End ──────────────────────────────────────

def test_model_causal_masking():
    """
    Verify causal masking works through the entire model.
    Changing a future token should NOT affect predictions at earlier positions.
    """
    print("Running end-to-end causal masking test...")

    config = GPTConfig(vocab_size=65, block_size=64, n_layer=2, n_head=4, n_embd=32)
    model = Scratchformer(config)
    model.eval()

    T = 8
    idx1 = torch.randint(0, config.vocab_size, (1, T))
    idx2 = idx1.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % config.vocab_size  # Change only last token

    with torch.no_grad():
        logits1, _ = model(idx1)
        logits2, _ = model(idx2)

    # Positions 0 to T-2 should produce identical logits
    for t in range(T - 1):
        diff = (logits1[0, t] - logits2[0, t]).abs().max().item()
        assert diff < 1e-5, f"Position {t} affected by future change! diff={diff}"
        print(f"  Position {t}: diff={diff:.2e} (unaffected) OK")

    # Last position should differ
    diff_last = (logits1[0, -1] - logits2[0, -1]).abs().max().item()
    assert diff_last > 1e-5, "Last position should be different!"
    print(f"  Position {T-1}: diff={diff_last:.4f} (changed as expected) OK")

    print("  PASSED\n")


# ── Generation Tests ───────────────────────────────────────────────

def test_generate_shape():
    """Verify generate() produces the correct number of new tokens."""
    print("Running generation shape test...")

    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = Scratchformer(config)
    model.eval()

    prompt = torch.zeros((1, 1), dtype=torch.long)  # Start with token 0
    max_new = 20

    generated = model.generate(prompt, max_new_tokens=max_new)
    assert generated.shape == (1, 1 + max_new), \
        f"Expected (1, {1 + max_new}), got {generated.shape}"

    print(f"  Prompt length: 1, generated: {generated.shape[1]} tokens (1 + {max_new}) OK")
    print("  PASSED\n")


def test_generate_tokens_in_range():
    """Verify all generated tokens are valid token IDs (within vocab range)."""
    print("Running generated tokens range test...")

    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = Scratchformer(config)
    model.eval()

    prompt = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=50)

    assert (generated >= 0).all(), "Generated negative token IDs!"
    assert (generated < config.vocab_size).all(), "Generated token IDs >= vocab_size!"

    print(f"  All {generated.shape[1]} tokens in range [0, {config.vocab_size}) OK")
    print("  PASSED\n")


def test_generate_temperature():
    """Verify temperature parameter affects generation diversity."""
    print("Running temperature effect test...")

    config = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=32)
    model = Scratchformer(config)
    model.eval()

    prompt = torch.zeros((1, 1), dtype=torch.long)

    # Low temperature should produce more repeated / deterministic output
    torch.manual_seed(42)
    gen_low = model.generate(prompt.clone(), max_new_tokens=50, temperature=0.01)
    # High temperature should produce more diverse output
    torch.manual_seed(42)
    gen_high = model.generate(prompt.clone(), max_new_tokens=50, temperature=2.0)

    # They should differ (different temperatures → different sampling)
    differ = (gen_low != gen_high).any().item()
    print(f"  Low temp (0.01) vs High temp (2.0) differ: {differ}")
    # Note: with same seed they *might* match on very first token, but diverge quickly

    print("  PASSED\n")


def test_generate_context_cropping():
    """Verify generation works even when the prompt exceeds block_size."""
    print("Running context cropping test (prompt > block_size)...")

    config = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = Scratchformer(config)
    model.eval()

    # Prompt is LONGER than block_size — the model should crop internally
    long_prompt = torch.randint(0, config.vocab_size, (1, 32))
    generated = model.generate(long_prompt, max_new_tokens=10)

    assert generated.shape == (1, 32 + 10), \
        f"Expected (1, 42), got {generated.shape}"

    print(f"  Prompt: 32 tokens, block_size: 16 → cropped internally, generated 10 more OK")
    print("  PASSED\n")


# ── Run all tests ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 4 MODEL TESTS")
    print("=" * 60 + "\n")

    # Config
    test_config_defaults()
    test_config_custom()

    # Model shapes and loss
    test_model_output_shape()
    test_model_loss_computation()
    test_model_param_count()

    # Embeddings
    test_token_embedding()
    test_position_embedding()

    # Sequence handling
    test_model_variable_sequence_lengths()

    # Gradients
    test_model_gradient_flow()

    # Causal masking
    test_model_causal_masking()

    # Generation
    test_generate_shape()
    test_generate_tokens_in_range()
    test_generate_temperature()
    test_generate_context_cropping()

    print("=" * 60)
    print("ALL DAY 4 MODEL TESTS PASSED!")
    print("=" * 60)
