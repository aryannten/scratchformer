import os
import json
from tokenizer import CharTokenizer

def test_tokenizer_roundtrip():
    print("Running tokenizer roundtrip test...")
    
    # 1. Simple test text
    test_text = "Hello, Scratchformer! 2026 World Cup is coming."
    tokenizer = CharTokenizer(text=test_text)
    
    # 2. Verify vocab size
    assert tokenizer.vocab_size > 0, "Vocabulary size must be greater than zero."
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # 3. Verify encoding/decoding roundtrip
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text, f"Roundtrip failed!\nOriginal: {test_text}\nDecoded:  {decoded}"
    print("OK Roundtrip encoding/decoding verified!")
    
    # 4. Save and load test
    temp_vocab_path = "tests/temp_vocab.json"
    tokenizer.save(temp_vocab_path)
    assert os.path.exists(temp_vocab_path), "Failed to save vocab file."
    
    loaded_tokenizer = CharTokenizer.load(temp_vocab_path)
    assert loaded_tokenizer.vocab_size == tokenizer.vocab_size, "Loaded tokenizer vocab size mismatch."
    assert loaded_tokenizer.chars == tokenizer.chars, "Loaded tokenizer unique chars mismatch."
    
    decoded_loaded = loaded_tokenizer.decode(encoded)
    assert decoded_loaded == test_text, f"Loaded tokenizer roundtrip failed!\nDecoded: {decoded_loaded}"
    print("OK Vocabulary saving and loading verified!")
    
    # Clean up
    if os.path.exists(temp_vocab_path):
        os.remove(temp_vocab_path)
        
    print("All tokenizer tests passed successfully!")

if __name__ == "__main__":
    test_tokenizer_roundtrip()
