import torch
import torch.nn.functional as F
from attention import Head, MultiHeadAttention


# ── Single Head Tests ──────────────────────────────────────────────

def test_head_shapes_and_masking():
    """Verify output shape and causal masking for a single attention head."""
    print("Running Head shapes and causal masking tests...")
    
    B, T, C = 2, 8, 32
    head_dim = 16
    block_size = 64
    
    head = Head(C, head_dim, block_size)
    x = torch.randn(B, T, C)
    
    # 1. Forward pass
    out = head(x)
    
    # 2. Verify shape
    assert out.shape == (B, T, head_dim), f"Incorrect output shape: {out.shape}, expected {(B, T, head_dim)}"
    print(f"  Output shape verified: {out.shape}")
    
    # 3. Verify causal masking manually
    with torch.no_grad():
        k = head.key(x)
        q = head.query(x)
        wei = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
        
        masked_wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
        probs = F.softmax(masked_wei, dim=-1)
        
        # Verify future positions are exactly 0.0
        for b in range(B):
            for i in range(T):
                for j in range(i + 1, T):
                    val = probs[b, i, j].item()
                    assert val == 0.0, f"Causal mask failed at batch {b}, row {i}, col {j}. Value: {val}"
                    
    print("  Causal masking verified (future tokens have 0 attention weights)!")
    print("  PASSED\n")


def test_head_different_sequence_lengths():
    """Verify Head works with T < block_size (which is the normal usage)."""
    print("Running Head with different sequence lengths...")
    
    B, C = 2, 32
    head_dim = 16
    block_size = 64
    head = Head(C, head_dim, block_size)
    
    for T in [1, 4, 16, 64]:
        x = torch.randn(B, T, C)
        out = head(x)
        assert out.shape == (B, T, head_dim), f"Failed for T={T}: {out.shape}"
        print(f"  T={T}: shape {out.shape} OK")
    
    print("  PASSED\n")


def test_head_gradients_flow():
    """Verify gradients flow back through the Head (important for training)."""
    print("Running Head gradient flow test...")
    
    B, T, C = 2, 8, 32
    head_dim = 16
    block_size = 64
    
    head = Head(C, head_dim, block_size)
    x = torch.randn(B, T, C, requires_grad=True)
    
    out = head(x)
    loss = out.sum()
    loss.backward()
    
    # Check that input gradients exist
    assert x.grad is not None, "No gradient on input x!"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape}"
    
    # Check that parameter gradients exist
    for name, param in head.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        print(f"  {name}: grad shape {param.grad.shape} OK")
    
    print("  PASSED\n")


# ── Multi-Head Attention Tests ─────────────────────────────────────

def test_multihead_output_shape():
    """Verify MultiHeadAttention preserves (B, T, embed_dim) shape."""
    print("Running MultiHeadAttention shape test...")
    
    B, T, C = 2, 8, 32
    num_heads = 4
    block_size = 64
    
    mha = MultiHeadAttention(C, num_heads, block_size)
    x = torch.randn(B, T, C)
    out = mha(x)
    
    assert out.shape == (B, T, C), f"Shape mismatch: expected {(B, T, C)}, got {out.shape}"
    print(f"  Output shape: {out.shape} OK")
    print("  PASSED\n")


def test_multihead_creates_correct_num_heads():
    """Verify the right number of Head modules are created."""
    print("Running MultiHeadAttention head count test...")
    
    C = 64
    for num_heads in [1, 2, 4, 8]:
        mha = MultiHeadAttention(C, num_heads, block_size=32)
        assert len(mha.heads) == num_heads, f"Expected {num_heads} heads, got {len(mha.heads)}"
        
        # Each head should have head_dim = C // num_heads
        expected_head_dim = C // num_heads
        for i, head in enumerate(mha.heads):
            # Check the key projection output dimension
            assert head.key.out_features == expected_head_dim, \
                f"Head {i}: expected head_dim={expected_head_dim}, got {head.key.out_features}"
        
        print(f"  num_heads={num_heads}, head_dim={expected_head_dim}: OK")
    
    print("  PASSED\n")


def test_multihead_embed_dim_not_divisible():
    """Verify MultiHeadAttention raises an error when embed_dim % num_heads != 0."""
    print("Running MultiHeadAttention divisibility assertion test...")
    
    try:
        mha = MultiHeadAttention(embed_dim=30, num_heads=4, block_size=32)
        assert False, "Should have raised an AssertionError!"
    except AssertionError:
        print("  Correctly raised AssertionError for embed_dim=30, num_heads=4")
    
    print("  PASSED\n")


def test_multihead_has_output_projection():
    """Verify the output projection layer exists and has the right shape."""
    print("Running MultiHeadAttention output projection test...")
    
    C = 64
    num_heads = 4
    mha = MultiHeadAttention(C, num_heads, block_size=32)
    
    # proj should be nn.Linear(embed_dim, embed_dim)
    assert hasattr(mha, 'proj'), "MultiHeadAttention missing 'proj' layer"
    assert mha.proj.in_features == C, f"proj input features: expected {C}, got {mha.proj.in_features}"
    assert mha.proj.out_features == C, f"proj output features: expected {C}, got {mha.proj.out_features}"
    print(f"  proj: Linear({mha.proj.in_features}, {mha.proj.out_features}) OK")
    print("  PASSED\n")


def test_multihead_causal_masking_preserved():
    """Verify causal masking still works when heads are combined."""
    print("Running MultiHeadAttention causal masking test...")
    
    B, T, C = 1, 4, 16
    num_heads = 4
    block_size = 8
    
    mha = MultiHeadAttention(C, num_heads, block_size)
    
    with torch.no_grad():
        # Create two inputs that differ only at position 3 (the last position)
        x1 = torch.randn(B, T, C)
        x2 = x1.clone()
        x2[:, 3, :] = torch.randn(C)  # change the last token
        
        out1 = mha(x1)
        out2 = mha(x2)
        
        # Positions 0, 1, 2 should NOT see position 3 (causal), so their outputs should be identical
        for t in range(T - 1):
            diff = (out1[:, t, :] - out2[:, t, :]).abs().max().item()
            assert diff < 1e-5, f"Position {t} changed (diff={diff}) even though only future token changed!"
            print(f"  Position {t}: diff={diff:.2e} (unchanged) OK")
        
        # Position 3 SHOULD differ
        diff_last = (out1[:, 3, :] - out2[:, 3, :]).abs().max().item()
        assert diff_last > 1e-5, f"Position 3 didn't change even though its input changed!"
        print(f"  Position 3: diff={diff_last:.4f} (changed as expected) OK")
    
    print("  PASSED\n")


def test_multihead_gradients_flow():
    """Verify gradients flow through MultiHeadAttention."""
    print("Running MultiHeadAttention gradient flow test...")
    
    B, T, C = 2, 8, 32
    num_heads = 4
    block_size = 64
    
    mha = MultiHeadAttention(C, num_heads, block_size)
    x = torch.randn(B, T, C, requires_grad=True)
    
    out = mha(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient on input x!"
    
    # Count parameters with gradients
    params_with_grad = 0
    params_total = 0
    for name, param in mha.named_parameters():
        params_total += 1
        if param.grad is not None:
            params_with_grad += 1
    
    assert params_with_grad == params_total, \
        f"Only {params_with_grad}/{params_total} parameters received gradients!"
    print(f"  All {params_total} parameters received gradients OK")
    print("  PASSED\n")


def test_multihead_parameter_count():
    """Verify total parameter count matches expectations."""
    print("Running MultiHeadAttention parameter count test...")
    
    C = 64
    num_heads = 4
    head_dim = C // num_heads  # 16
    
    mha = MultiHeadAttention(C, num_heads, block_size=32)
    
    total_params = sum(p.numel() for p in mha.parameters())
    
    # Expected:
    # Each head: 3 linear layers (Q, K, V) of shape (C, head_dim) with no bias = 3 * C * head_dim
    # Projection: (C, C) + bias of C
    # Total = num_heads * 3 * C * head_dim + C * C + C
    expected = num_heads * 3 * C * head_dim + C * C + C
    
    assert total_params == expected, f"Parameter count: expected {expected}, got {total_params}"
    print(f"  Total parameters: {total_params} (expected {expected}) OK")
    print("  PASSED\n")


# ── Run all tests ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 3 ATTENTION TESTS")
    print("=" * 60 + "\n")
    
    # Single Head
    test_head_shapes_and_masking()
    test_head_different_sequence_lengths()
    test_head_gradients_flow()
    
    # Multi-Head
    test_multihead_output_shape()
    test_multihead_creates_correct_num_heads()
    test_multihead_embed_dim_not_divisible()
    test_multihead_has_output_projection()
    test_multihead_causal_masking_preserved()
    test_multihead_gradients_flow()
    test_multihead_parameter_count()
    
    print("=" * 60)
    print("ALL ATTENTION TESTS PASSED!")
    print("=" * 60)
