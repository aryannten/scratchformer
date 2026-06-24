import os
import requests
import torch
from tokenizer import CharTokenizer

def download_file(url, dest_path):
    """Download a file from a URL to a local destination path."""
    print(f"Downloading {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(response.content)
    print("Download complete!")

def main():
    # Paths
    raw_dir = os.path.join("data", "raw")
    prepared_dir = os.path.join("data", "prepared")
    
    shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    shakespeare_raw_path = os.path.join(raw_dir, "tinyshakespeare.txt")
    
    # 1. Download Tiny Shakespeare if not present
    if not os.path.exists(shakespeare_raw_path):
        download_file(shakespeare_url, shakespeare_raw_path)
    
    # 2. Read the raw text
    with open(shakespeare_raw_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset loaded: {len(text):,} characters.")
    
    # 3. Create tokenizer and save vocab
    tokenizer = CharTokenizer(text=text)
    vocab_path = os.path.join(prepared_dir, "vocab.json")
    tokenizer.save(vocab_path)
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters. Vocab saved to {vocab_path}.")
    
    # 4. Train/val split (90% train, 10% val)
    n = len(text)
    train_text = text[:int(n*0.9)]
    val_text = text[int(n*0.9):]
    
    # 5. Encode to integers
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    
    print(f"Train set has {len(train_ids):,} tokens.")
    print(f"Validation set has {len(val_ids):,} tokens.")
    
    # 6. Convert to PyTorch tensors and save
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    val_tensor = torch.tensor(val_ids, dtype=torch.long)
    
    train_dest = os.path.join(prepared_dir, "train.pt")
    val_dest = os.path.join(prepared_dir, "val.pt")
    
    torch.save(train_tensor, train_dest)
    torch.save(val_tensor, val_dest)
    
    print(f"Tensors saved successfully to {train_dest} and {val_dest}!")

if __name__ == "__main__":
    main()
