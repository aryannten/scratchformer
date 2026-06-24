import json
import os

class CharTokenizer:
    def __init__(self, text: str = None, chars: list = None):
        """
        Initialize the character-level tokenizer.
        If `text` is provided, build vocabulary from it.
        If `chars` is provided directly, use it to build the vocabulary.
        """
        if text is not None:
            self.chars = sorted(list(set(text)))
        elif chars is not None:
            self.chars = sorted(list(chars))
        else:
            self.chars = []
            
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """Convert a string to a list of integer token IDs."""
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of integer token IDs to a string."""
        return ''.join([self.itos[t] for t in tokens if t in self.itos])

    def save(self, filepath: str):
        """Save the tokenizer vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.chars, f, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str):
        """Load the tokenizer vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            chars = json.load(f)
        return cls(chars=chars)
