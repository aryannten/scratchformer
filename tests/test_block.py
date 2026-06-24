import torch
from block import TransformerBlock

def test_block_shape_preservation():
    print("Running TransformerBlock shape preservation tests...")
    
    B, T, C = 2, 8, 32
    num_heads = 4
    block_size = 64
    
    # Initialize the transformer block
    block = TransformerBlock(embed_dim=C, num_heads=num_heads, block_size=block_size)
    x = torch.randn(B, T, C)
    
    # Forward pass
    out = block(x)
    
    # Verify shape is preserved
    assert out.shape == (B, T, C), f"Shape not preserved! Expected {(B, T, C)}, got {out.shape}"
    print(f"TransformerBlock shape preservation verified: {out.shape}")
    print("All transformer block tests passed successfully!")

if __name__ == "__main__":
    test_block_shape_preservation()
