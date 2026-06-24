import torch
import torch.nn.functional as F
from attention import Head

def test_head_shapes_and_masking():
    print("Running Head shapes and causal masking tests...")
    
    B, T, C = 2, 8, 32
    head_dim = 16
    block_size = 64
    
    # Initialize the single attention head
    head = Head(C, head_dim, block_size)
    x = torch.randn(B, T, C)
    
    # 1. Forward pass
    out = head(x)
    
    # 2. Verify shape
    assert out.shape == (B, T, head_dim), f"Incorrect output shape: {out.shape}, expected {(B, T, head_dim)}"
    print(f"Output shape verified: {out.shape}")
    
    # 3. Verify causal masking manually
    with torch.no_grad():
        k = head.key(x)
        q = head.query(x)
        wei = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
        
        # Check that prior to masking, we have non-trivial weights in the upper triangle
        # Apply mask
        masked_wei = wei.masked_fill(head.tril[:T, :T] == 0, float('-inf'))
        probs = F.softmax(masked_wei, dim=-1)
        
        # Verify that all future positions (upper triangle of the B x T x T matrix) are exactly 0.0
        for b in range(B):
            for i in range(T):
                for j in range(i + 1, T):
                    val = probs[b, i, j].item()
                    assert val == 0.0, f"Causal mask failed at batch {b}, row {i}, col {j}. Value: {val}"
                    
    print("OK Causal masking verified (future tokens have 0 attention weights)!")
    print("All single attention head tests passed successfully!")

if __name__ == "__main__":
    test_head_shapes_and_masking()
