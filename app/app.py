import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from collections import Counter

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdu20.txt')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# Data Processing Functions
def load_data(file_path):
    english_sentences = []
    urdu_sentences = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                english_sentences.append(parts[0].strip())
                urdu_sentences.append(parts[1].strip())
    
    return list(zip(english_sentences, urdu_sentences))

def simple_tokenize(text):
    return text.split()

def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for sentence in sentences:
        tokens = simple_tokenize(sentence)
        counter.update(tokens)
    
    vocab = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
    idx = len(vocab)
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


# Model Architecture: Seq2Seq with Additive (Bahdanau) Attention
class Encoder(nn.Module):   
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden


class AdditiveAttention(nn.Module):   
    def __init__(self, hid_dim):
        super().__init__()
        self.v = nn.Linear(hid_dim, 1, bias=False)
        self.W = nn.Linear(hid_dim, hid_dim)      # W2: for decoder hidden state
        self.U = nn.Linear(hid_dim * 2, hid_dim)  # W1: for encoder outputs
    
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Additive attention: v^T * tanh(W1*h + W2*s)
        energy = self.v(torch.tanh(self.W(hidden) + self.U(encoder_outputs))).squeeze(2)
        energy = energy.masked_fill(mask, -1e10)
        attention = F.softmax(energy, dim=1)
        
        return attention


class GeneralAttention(nn.Module):  
    def __init__(self, hid_dim):
        super().__init__()
        # Project encoder outputs (hid_dim*2) to decoder hidden size (hid_dim)
        self.W = nn.Linear(hid_dim * 2, hid_dim, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, src_len, hid*2]
        
        # Project encoder outputs to match decoder hidden dimension
        projected = self.W(encoder_outputs)  # [batch, src_len, hid_dim]
        
        # General attention: s^T * W * h
        hidden = hidden.unsqueeze(2)  # [batch, hid_dim, 1]
        energy = torch.bmm(projected, hidden).squeeze(2)  # [batch, src_len]
        
        energy = energy.masked_fill(mask, -1e10)
        attention = F.softmax(energy, dim=1)
        
        return attention


class Decoder(nn.Module):    
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc(torch.cat((embedded, output, weighted), dim=1))
        
        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2SeqAttention(nn.Module):    
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src == self.src_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        
        input = trg[0, :]
        mask = self.create_mask(src)
        
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[t] = output
            top1 = output.argmax(1)
            input = top1
            
        return outputs, None


# Translation Functions
def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    
    # Tokenize and convert to indices
    tokens = sentence.split()
    ids = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    ids = [src_vocab['<sos>']] + ids + [src_vocab['<eos>']]
    
    # Convert to tensor
    src_tensor = torch.LongTensor(ids).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(ids)]).to(device)
    
    # Create reverse vocabulary
    idx_to_word = {idx: word for word, idx in trg_vocab.items()}
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)
    
    mask = model.create_mask(src_tensor)
    trg_idx = trg_vocab['<sos>']
    translated_tokens = []
    
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_idx]).to(device)
        
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
        
        pred_token = output.argmax(1).item()
        translated_tokens.append(idx_to_word.get(pred_token, '<unk>'))
        
        if pred_token == trg_vocab['<eos>']:
            break
        
        trg_idx = pred_token
    
    # Remove <eos> if present
    if '<eos>' in translated_tokens:
        translated_tokens = translated_tokens[:translated_tokens.index('<eos>')]
    
    return ' '.join(translated_tokens)


# Model Loading
def load_vocabularies():
    print("Loading data and building vocabularies...")
    data = load_data(DATA_PATH)
    
    # Use same seed as training for consistent vocabulary
    import random
    random.seed(42)
    random.shuffle(data)
    
    train_size = int(len(data) * 0.7)
    train_data = data[:train_size]
    
    train_src = [pair[0] for pair in train_data]
    train_trg = [pair[1] for pair in train_data]
    
    src_vocab = build_vocab(train_src, min_freq=2)
    trg_vocab = build_vocab(train_trg, min_freq=2)
    
    print(f"  English vocab: {len(src_vocab)} tokens")
    print(f"  Urdu vocab: {len(trg_vocab)} tokens")
    
    return src_vocab, trg_vocab


def load_model(src_vocab, trg_vocab, attention_type='additive'):
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    DROPOUT = 0.5
    PAD_IDX = src_vocab['<pad>']
    
    # Select attention mechanism
    if attention_type == 'general':
        attn = GeneralAttention(HID_DIM)
        model_path = os.path.join(MODEL_DIR, 'general_attention_model.pt')
    else:
        attn = AdditiveAttention(HID_DIM)
        model_path = os.path.join(MODEL_DIR, 'attention_model.pt')
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT, attn)
    model = Seq2SeqAttention(enc, dec, PAD_IDX, DEVICE).to(DEVICE)
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print(f"Using untrained model weights.")
    
    model.eval()
    return model


# Initialize Models
print("English to Urdu Machine Translation")
print("NLU Assignment 3 - Task 4")
print(f"Device: {DEVICE}")
print()

# Load vocabularies
src_vocab, trg_vocab = load_vocabularies()

# Load model (using General Attention - best performing based on experiments)
print("\nLoading General Attention model (best performer)...")
model_general = load_model(src_vocab, trg_vocab, 'general')

# Also load Additive Attention for comparison
print("\nLoading Additive Attention model...")
model_additive = load_model(src_vocab, trg_vocab, 'additive')
has_additive = True


# Gradio Interface
def translate(text, attention_type):
    if not text.strip():
        return "Please enter a sentence to translate."
    
    # Select model based on attention type
    if attention_type == "Additive Attention" and has_additive:
        model = model_additive
    else:
        model = model_general  # Default to General (best performer)
    
    try:
        translation = translate_sentence(text, src_vocab, trg_vocab, model, DEVICE)
        return translation
    except Exception as e:
        return f"Translation error: {str(e)}"

# Build Gradio interface
with gr.Blocks(title="English to Urdu Translator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # English to Urdu Machine Translation
        
    This web application demonstrates Neural Machine Translation using Seq2Seq models 
    with attention mechanisms trained on the Tatoeba English-Urdu parallel corpus.
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="English Input",
                placeholder="Enter an English sentence...",
                lines=3
            )
            
            attention_choice = gr.Radio(
                choices=["General Attention", "Additive Attention"],
                value="General Attention",
                label="Attention Mechanism",
                info="General Attention achieved best performance (PPL: 1.264)"
            )
            
            translate_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Urdu Translation",
                lines=3,
                rtl=True  # Right-to-left for Urdu
            )
    
    # Example sentences as clickable buttons
    gr.Markdown("### Try an Example")
    
    example_sentences = [
        "I love you.",
        "How are you?",
        "Good morning.",
        "What is your name?",
        "I am a student.",
        "Thank you very much."
    ]
    
    with gr.Row():
        example_buttons = []
        for sentence in example_sentences:
            btn = gr.Button(sentence, size="sm", variant="secondary")
            example_buttons.append(btn)
    
    # Connect example buttons to fill input
    for btn in example_buttons:
        btn.click(
            fn=lambda x: x,
            inputs=[btn],
            outputs=[input_text]
        )
    
    # Connect translate button to function
    translate_btn.click(
        fn=translate,
        inputs=[input_text, attention_choice],
        outputs=output_text
    )
    
    input_text.submit(
        fn=translate,
        inputs=[input_text, attention_choice],
        outputs=output_text
    )

# Launch Application
if __name__ == "__main__":
    print("Starting Gradio Web Application...")
    demo.launch(share=False)
