import gradio as gr
import torch
import os
from tokenizer import CharTokenizer
from model import Scratchformer

# Configuration
CHECKPOINT_PATH = "checkpoints/best.pt"
VOCAB_PATH = "checkpoints/custom_vocab.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer():
    print("Loading tokenizer...")
    vocab_path = VOCAB_PATH
    if not os.path.exists(vocab_path):
        # Fallback to data dir if running in different folder
        vocab_path = "data/prepared/custom_vocab.json"
        
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_PATH} or {vocab_path}")

    tokenizer = CharTokenizer.load(vocab_path)

    print(f"Loading model from {CHECKPOINT_PATH} to {DEVICE}...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}. Please download it from Colab or ensure it is in the correct directory.")

    # Note: we need weights_only=False because the checkpoint contains the GPTConfig dataclass object
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    model_config = checkpoint['model_config']
    model = Scratchformer(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer

try:
    model, tokenizer = load_model_and_tokenizer()
except Exception as e:
    print(f"Warning: {e}")
    model, tokenizer = None, None

def generate_story(prompt, max_tokens, temperature, top_k):
    if model is None or tokenizer is None:
        return "Error: Model or Tokenizer not loaded correctly. Check the console for errors."
        
    if prompt.strip():
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=int(max_tokens), temperature=float(temperature), top_k=int(top_k))
    
    return tokenizer.decode(generated[0].tolist())

# Gradio Interface
css = """
body {
    background-color: #f0f4f8;
}
.gradio-container {
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), css=css) as demo:
    gr.Markdown("# ⚽ Scratchformer: AI Football Narratives")
    gr.Markdown("This AI was built **completely from scratch** and trained on a custom Wikipedia corpus of FIFA World Cup history. Type a prompt to see how it writes football narratives!")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. In the 1970 FIFA World Cup final...",
                lines=3
            )
            
            with gr.Row():
                max_tokens = gr.Slider(minimum=50, maximum=1000, value=300, step=10, label="Max Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.1, label="Temperature (Creativity)")
                top_k = gr.Slider(minimum=1, maximum=100, value=40, step=1, label="Top-K")
            
            generate_btn = gr.Button("Generate Story 🚀", variant="primary")
            
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="Generated Narrative",
                lines=12,
                interactive=False
            )
            
    # Connect UI
    generate_btn.click(
        fn=generate_story,
        inputs=[prompt, max_tokens, temperature, top_k],
        outputs=output
    )
    
    gr.Examples(
        examples=[
            ["The 1970 FIFA World Cup in Mexico", 300, 0.8, 40],
            ["Diego Maradona scored a memorable", 300, 0.75, 40],
            ["In the final match of the World Cup,", 250, 0.8, 40],
            ["Pele is widely considered", 300, 0.7, 40],
            ["Total Football was", 300, 0.8, 40]
        ],
        inputs=[prompt, max_tokens, temperature, top_k]
    )

if __name__ == "__main__":
    demo.launch(share=False)
