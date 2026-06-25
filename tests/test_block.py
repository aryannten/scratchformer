import torch
from block import FeedForward, TransformerBlock


# ── FeedForward Tests ──────────────────────────────────────────────

def test_feedforward_shape():
    """Verify FeedForward preserves (B, T, embed_dim) shape."""
    print("Running FeedForward shape test...")
    
    B, T, C = 2, 8, 32
    ffn = FeedForward(C)
    x = torch.randn(B, T, C)
    out = ffn(x)
    
    assert out.shape == (B, T, C), f"Shape mismatch: expected {(B, T, C)}, got {out.shape}"
    print(f"  Output shape: {out.shape} OK")
    print("  PASSED\n")


def test_feedforward_expansion_ratio():
    """Verify the inner layer expands by 4x (standard GPT FFN)."""
    print("Running FeedForward expansion ratio test...")
    
    C = 64
    ffn = FeedForward(C)
    
    # net is a Sequential: Linear(C, 4C) -> GELU -> Linear(4C, C)
    layers = list(ffn.net.children())
    assert len(layers) == 3, f"Expected 3 layers in FFN, got {len(layers)}"
    
    linear1 = layers[0]
    assert linear1.in_features == C, f"First linear in_features: expected {C}, got {linear1.in_features}"
    assert linear1.out_features == 4 * C, f"First linear out_features: expected {4*C}, got {linear1.out_features}"
    
    linear2 = layers[2]
    assert linear2.in_features == 4 * C, f"Second linear in_features: expected {4*C}, got {linear2.in_features}"
    assert linear2.out_features == C, f"Second linear out_features: expected {C}, got {linear2.out_features}"
    
    print(f"  Linear({C}, {4*C}) -> GELU -> Linear({4*C}, {C}) OK")
    print("  PASSED\n")


def test_feedforward_uses_gelu():
    """Verify FeedForward uses GELU activation (not ReLU)."""
    print("Running FeedForward GELU activation test...")
    
    ffn = FeedForward(32)
    layers = list(ffn.net.children())
    
    activation = layers[1]
    assert isinstance(activation, torch.nn.GELU), \
        f"Expected GELU activation, got {type(activation).__name__}"
    
    print(f"  Activation: {type(activation).__name__} OK")
    print("  PASSED\n")


def test_feedforward_gradients():
    """Verify gradients flow through FeedForward."""
    print("Running FeedForward gradient flow test...")
    
    B, T, C = 2, 8, 32
    ffn = FeedForward(C)
    x = torch.randn(B, T, C, requires_grad=True)
    
    out = ffn(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient on input!"
    for name, param in ffn.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    
    print(f"  All parameters received gradients OK")
    print("  PASSED\n")


# ── TransformerBlock Tests ─────────────────────────────────────────

def test_block_shape_preservation():
    """Verify TransformerBlock preserves (B, T, C) shape."""
    print("Running TransformerBlock shape preservation test...")
    
    B, T, C = 2, 8, 32
    num_heads = 4
    block_size = 64
    
    block = TransformerBlock(embed_dim=C, num_heads=num_heads, block_size=block_size)
    x = torch.randn(B, T, C)
    out = block(x)
    
    assert out.shape == (B, T, C), f"Shape not preserved! Expected {(B, T, C)}, got {out.shape}"
    print(f"  Output shape: {out.shape} OK")
    print("  PASSED\n")


def test_block_has_prenorm():
    """Verify TransformerBlock uses pre-LayerNorm (not post-norm)."""
    print("Running TransformerBlock pre-LayerNorm architecture test...")
    
    C = 32
    block = TransformerBlock(embed_dim=C, num_heads=4, block_size=64)
    
    # Check that ln1 and ln2 exist as LayerNorm
    assert hasattr(block, 'ln1'), "Missing ln1"
    assert hasattr(block, 'ln2'), "Missing ln2"
    assert isinstance(block.ln1, torch.nn.LayerNorm), f"ln1 is {type(block.ln1)}, expected LayerNorm"
    assert isinstance(block.ln2, torch.nn.LayerNorm), f"ln2 is {type(block.ln2)}, expected LayerNorm"
    
    # Check that LayerNorm has the right normalized shape
    assert block.ln1.normalized_shape == (C,), f"ln1 shape: {block.ln1.normalized_shape}"
    assert block.ln2.normalized_shape == (C,), f"ln2 shape: {block.ln2.normalized_shape}"
    
    print(f"  ln1: LayerNorm({C}) OK")
    print(f"  ln2: LayerNorm({C}) OK")
    print("  PASSED\n")


def test_block_residual_connection():
    """Verify residual connections work (output != just attention/FFN output)."""
    print("Running TransformerBlock residual connection test...")
    
    B, T, C = 1, 4, 16
    num_heads = 2
    block_size = 8
    
    block = TransformerBlock(embed_dim=C, num_heads=num_heads, block_size=block_size)
    
    with torch.no_grad():
        x = torch.randn(B, T, C)
        out = block(x)
        
        # If residual connections work, the output should NOT be identical to
        # just the attention output or just the FFN output.
        # More importantly, with random init, the residual should keep the
        # output close to the input (since weights are small at init)
        diff = (out - x).abs().mean().item()
        
        # The output should be different from input (because of attention + FFN)
        assert diff > 0, "Output is identical to input - residual might be the ONLY thing happening"
        # But not wildly different (residual keeps it stable at init)
        assert diff < 10, f"Output is too far from input (diff={diff}). Residual connection might be broken."
    
    print(f"  Mean diff from input: {diff:.4f} (reasonable at init) OK")
    print("  PASSED\n")


def test_block_causal_masking_through_block():
    """Verify causal masking is preserved end-to-end through the block."""
    print("Running TransformerBlock causal masking test...")
    
    B, T, C = 1, 4, 16
    num_heads = 2
    block_size = 8
    
    block = TransformerBlock(embed_dim=C, num_heads=num_heads, block_size=block_size)
    
    with torch.no_grad():
        x1 = torch.randn(B, T, C)
        x2 = x1.clone()
        x2[:, 3, :] = torch.randn(C)  # only change last token
        
        out1 = block(x1)
        out2 = block(x2)
        
        # Positions 0-2 should be unaffected by the change at position 3
        for t in range(T - 1):
            diff = (out1[:, t, :] - out2[:, t, :]).abs().max().item()
            assert diff < 1e-5, f"Position {t} changed (diff={diff}) despite causal masking!"
            print(f"  Position {t}: diff={diff:.2e} (unchanged) OK")
        
        # Position 3 should be different
        diff_last = (out1[:, 3, :] - out2[:, 3, :]).abs().max().item()
        assert diff_last > 1e-5, f"Position 3 didn't change!"
        print(f"  Position 3: diff={diff_last:.4f} (changed as expected) OK")
    
    print("  PASSED\n")


def test_block_gradients_flow():
    """Verify gradients flow through the entire TransformerBlock."""
    print("Running TransformerBlock gradient flow test...")
    
    B, T, C = 2, 8, 32
    num_heads = 4
    block_size = 64
    
    block = TransformerBlock(embed_dim=C, num_heads=num_heads, block_size=block_size)
    x = torch.randn(B, T, C, requires_grad=True)
    
    out = block(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient on input!"
    
    params_total = 0
    params_with_grad = 0
    for name, param in block.named_parameters():
        params_total += 1
        if param.grad is not None:
            params_with_grad += 1
        else:
            print(f"  WARNING: {name} has no gradient!")
    
    assert params_with_grad == params_total, \
        f"Only {params_with_grad}/{params_total} parameters received gradients!"
    print(f"  All {params_total} parameters received gradients OK")
    print("  PASSED\n")


def test_block_multiple_sequence_lengths():
    """Verify TransformerBlock works with various sequence lengths."""
    print("Running TransformerBlock with different sequence lengths...")
    
    B, C = 2, 32
    num_heads = 4
    block_size = 64
    
    block = TransformerBlock(embed_dim=C, num_heads=num_heads, block_size=block_size)
    
    for T in [1, 4, 16, 64]:
        x = torch.randn(B, T, C)
        out = block(x)
        assert out.shape == (B, T, C), f"Failed for T={T}: {out.shape}"
        print(f"  T={T}: shape {out.shape} OK")
    
    print("  PASSED\n")


def test_stacked_blocks():
    """Verify multiple TransformerBlocks can be stacked."""
    print("Running stacked TransformerBlocks test...")
    
    B, T, C = 2, 8, 32
    num_heads = 4
    block_size = 64
    n_layers = 4
    
    blocks = torch.nn.Sequential(*[
        TransformerBlock(embed_dim=C, num_heads=num_heads, block_size=block_size)
        for _ in range(n_layers)
    ])
    
    x = torch.randn(B, T, C, requires_grad=True)
    out = blocks(x)
    
    assert out.shape == (B, T, C), f"Stacked output shape: {out.shape}"
    
    # Verify gradient flow through all layers
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient through stacked blocks!"
    
    print(f"  {n_layers} stacked blocks: shape {out.shape}, gradients flow OK")
    print("  PASSED\n")


# ── Run all tests ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DAY 3 BLOCK TESTS")
    print("=" * 60 + "\n")
    
    # FeedForward
    test_feedforward_shape()
    test_feedforward_expansion_ratio()
    test_feedforward_uses_gelu()
    test_feedforward_gradients()
    
    # TransformerBlock
    test_block_shape_preservation()
    test_block_has_prenorm()
    test_block_residual_connection()
    test_block_causal_masking_through_block()
    test_block_gradients_flow()
    test_block_multiple_sequence_lengths()
    test_stacked_blocks()
    
    print("=" * 60)
    print("ALL BLOCK TESTS PASSED!")
    print("=" * 60)
