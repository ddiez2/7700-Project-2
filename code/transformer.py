import os
import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class TokenizedDataset(Dataset):
    def __init__(self, filepath, sp=None, max_seq_length=512):
        self.samples = []
        self.max_seq_length = max_seq_length
        f = open(filepath, "r", encoding="utf-8")
        for line in f:
            data = json.loads(line.strip())
            tokens = data["tokens"]
            
            if tokens and type(tokens[0]) == str:
                if tokens[0].isdigit():
                    tokens = [int(t) for t in tokens]
                else:
                    tokens = [sp.piece_to_id(t) for t in tokens]
            
            if len(tokens) <= 1:
                continue
                
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                
            self.samples.append(tokens)
        f.close()
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        toks = self.samples[idx]
        inputs = toks[:-1]
        targets = toks[1:]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def collate_fn(batch, pad_token=0):
    inputs = []
    targets = []
    for i in range(len(batch)):
        inputs.append(batch[i][0])
        targets.append(batch[i][1])
    
    max_len = 0
    for seq in inputs:
        if seq.size(0) > max_len:
            max_len = seq.size(0)
    
    padded_inputs = []
    padded_targets = []
    for i in range(len(inputs)):
        curr_len = inputs[i].size(0)
        padding_size = max_len - curr_len
        if padding_size > 0:
            padding = torch.ones(padding_size, dtype=torch.long) * pad_token
            padded_input = torch.cat([inputs[i], padding])
            padded_target = torch.cat([targets[i], padding])
        else:
            padded_input = inputs[i]
            padded_target = targets[i]
        
        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=6,
                 max_seq_length=512, dropout=0.1, eos_token=3):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.eos_token = eos_token
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.register_buffer("positional_encoding", self.create_sinusoidal_encoding(max_seq_length, d_model))
        self.pos_dropout = nn.Dropout(p=dropout)
        
        self.prenorm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.final_norm = nn.LayerNorm(d_model)
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight
    
    def create_sinusoidal_encoding(self, max_len, d_model):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe * 0.1
    
    def forward(self, x, temperature=1.0, sample=False):
        batch_size, seq_len = x.size()
        
        attention_mask = self.create_padding_mask(x)
        key_padding_mask = (x == 0)
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.pos_dropout(x)
        
        x = self.prenorm(x)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        x = self.transformer_encoder(
            x, 
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )
        
        x = self.final_norm(x)
        
        logits = self.fc(x)
        
        if sample:
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token.squeeze(1)
        else:
            return logits
    
    def create_padding_mask(self, x):
        return x == 0
    
    def prompt(self, text, sp, device, max_gen_length=None, temperature=1.0):
        if max_gen_length is None:
            max_gen_length = self.max_seq_length
        token_ids = sp.encode(text, out_type=int)
        generated = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            for _ in range(max_gen_length - generated.size(1)):
                next_token = self.forward(generated, temperature=temperature, sample=True)
                if next_token.item() == self.eos_token:
                    break
                next_token = next_token.unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
        
        generated_ids = generated.squeeze(0).tolist()
        if self.eos_token in generated_ids:
            eos_pos = generated_ids.index(self.eos_token)
            generated_ids = generated_ids[:eos_pos]
        
        return sp.decode_ids(generated_ids)

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        non_pad_mask = (targets != 0)
        num_tokens = non_pad_mask.sum().item()
        
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(inputs, sample=False)
                loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs, sample=False)
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
    return total_loss / total_tokens if total_tokens > 0 else float('inf')

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            non_pad_mask = (targets != 0)
            num_tokens = non_pad_mask.sum().item()
            
            logits = model(inputs, sample=False)
            loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def evaluate_bleu(model, test_filepath, sp, device, max_gen_length=None, temperature=1.0, eval_percentage=100):
    references = []
    hypotheses = []
    
    with open(test_filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    test_size = len(all_lines)
    if eval_percentage < 100:
        sample_size = max(1, int(test_size * eval_percentage / 100))
        import random
        random.seed(42)
        sampled_lines = random.sample(all_lines, sample_size)
        print(f"BLEU evaluation on {eval_percentage}% of test data ({sample_size}/{test_size} examples)")
    else:
        sampled_lines = all_lines
        print(f"BLEU evaluation on 100% of test data ({test_size} examples)")
    
    def predict_next_token(text):
        token_ids = sp.encode(text, out_type=int)
        input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            next_token_id = model.forward(input_tensor, temperature=temperature, sample=True).item()
            next_token_text = sp.decode_ids([next_token_id])
            return next_token_text
    
    for line in tqdm(sampled_lines, desc="BLEU Evaluating", leave=False):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]
        
        reference_first_token = reference_text.split()[0] if reference_text else ""
        
        try:
            predicted_token = predict_next_token(prompt_text)
            predicted_token = predicted_token.strip()
        except Exception as e:
            print(f"Error predicting next token: {e}")
            predicted_token = ""
        
        if len(hypotheses) < 3:
            print(f"\nSample {len(hypotheses) + 1}:")
            print(f"Prompt: {prompt_text}")
            print(f"Reference token: '{reference_first_token}'")
            print(f"Predicted token: '{predicted_token}'")
            print("-" * 40)
        
        references.append([reference_first_token])
        hypotheses.append([predicted_token])
    
    smoothing = SmoothingFunction().method1
    
    bleu_1 = corpus_bleu(
        references, 
        hypotheses, 
        weights=(1, 0, 0, 0), 
        smoothing_function=smoothing
    )
    
    correct = 0
    total = len(references)
    for ref, hyp in zip(references, hypotheses):
        if hyp[0] == ref[0]:
            correct += 1
    accuracy = (correct / total) * 100
    
    print(f"\nBLEU-1 Score: {bleu_1:.4f}")
    print(f"Exact match accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")
    
    return bleu_1

def generate_completions(model, test_filepath, sp, device, max_gen_length=None, temperature=1.0, num_samples=5):
    with open(test_filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    import random
    random.seed(42)
    sample_lines = random.sample(all_lines, min(num_samples, len(all_lines)))
    
    print(f"\n--- Full Text Generation Examples (temperature={temperature}) ---")
    
    for i, line in enumerate(sample_lines):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]
        
        generated_text = model.prompt(prompt_text, sp, device, max_gen_length=max_gen_length, temperature=temperature)
        
        if prompt_text in generated_text:
            generated_completion = generated_text[len(prompt_text):].strip()
        else:
            generated_completion = generated_text.strip()
        
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompt_text}")
        print(f"Reference: {reference_text}")
        print(f"Generated: {generated_completion}")
        print("-" * 40)

def plot_loss_curves(train_losses, val_losses, save_path="transformer_loss_curve.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="../data/train_tokenized.jsonl",
                        help="Path to tokenized training JSONL file.")
    parser.add_argument("--test_file", type=str, default="../data/test_tokenized.jsonl",
                        help="Path to tokenized test JSONL file for perplexity evaluation.")
    parser.add_argument("--test_orig_file", type=str, default="../data/test.jsonl",
                        help="Path to original test JSONL (with prompt and completion) for BLEU evaluation.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Learning rate warmup steps")
    args = parser.parse_args()
    
    vocab_size = 10000
    lr = args.lr
    patience = 15
    max_seq_length = 512
    model_save_path = "best_transformer_model.pt"
    
    d_model = 512
    nhead = 8
    num_layers = 6
    dropout = 0.1
    clip_grad_norm = 3.0
    weight_decay = 0.01  
    eval_every = 2
    
    train_losses = []
    val_losses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sp = spm.SentencePieceProcessor()
    sp_model_path = "../tokenizer/tokenizer.model"
    sp.load(sp_model_path)
    
    train_full_dataset = TokenizedDataset(args.train_file, sp=sp, max_seq_length=max_seq_length)
    
    train_size = int(0.9 * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, val_size])
    
    test_dataset = TokenizedDataset(args.test_file, sp=sp, max_seq_length=max_seq_length)
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model with size 512 for bigger capacity
    model = TransformerLM(
        vocab_size=vocab_size, 
        d_model=d_model,  # 512 dim embeddings for richer representation
        nhead=nhead,      # 8 attention heads for parallel feature extraction
        num_layers=num_layers,  # 6 layers strikes balance between depth and efficiency
        max_seq_length=max_seq_length,  
        dropout=dropout,  # 0.1 dropout prevents overfitting
        eos_token=3
    )
    
    # Custom weight initialization for better convergence
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            # Kaiming init for linear layers helps with gradient flow
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal init for embeddings with small std to start with reasonable values
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard init for layer norm
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    model.apply(_initialize_weights)
    model.to(device)
    
    # Using 0 as padding token to ignore in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
    
    # AdamW with weight decay for regularization
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, 0.98),  # Higher beta2 for more stable updates with transformer
        eps=1e-8, 
        weight_decay=weight_decay  # L2 regularization
    )
    
    # Calculate total steps for learning rate scheduling
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    # Learning rate warmup helps stabilize early training
    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    best_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    
    total_training_start = time.time()
    print("Starting training...")
    
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            non_pad_mask = (targets != 0)
            num_tokens = non_pad_mask.sum().item()
            
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    logits = model(inputs, sample=False)
                    loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                    loss = loss / args.accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                logits = model(inputs, sample=False)
                loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                loss = loss / args.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            
            train_loss += (loss.item() * args.accumulation_steps * num_tokens)
            total_tokens += num_tokens
            
            progress_bar.set_postfix({"loss": loss.item() * args.accumulation_steps, 
                                     "lr": optimizer.param_groups[0]['lr']})
        
        if (batch_idx + 1) % args.accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
        
        train_loss = train_loss / total_tokens if total_tokens > 0 else float('inf')
        train_losses.append(train_loss)
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Time: {epoch_elapsed:.2f}s")
        
        if epoch % eval_every == 0:
            val_loss, val_ppl = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'hyperparams': {
                        'd_model': d_model,
                        'nhead': nhead,
                        'num_layers': num_layers,
                        'dropout': dropout,
                        'vocab_size': vocab_size
                    }
                }, model_save_path)
                print("Model improved; saving model.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            val_losses.append(None)
    
    total_training_elapsed = time.time() - total_training_start
    print(f"Total training time: {total_training_elapsed:.2f}s")
    
    val_losses_cleaned = []
    train_losses_for_plot = []
    for i, val_loss in enumerate(val_losses):
        if val_loss is not None:
            val_losses_cleaned.append(val_loss)
            train_losses_for_plot.append(train_losses[i])
    
    if len(val_losses_cleaned) > 0:
        plot_loss_curves(train_losses_for_plot, val_losses_cleaned)
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss_curve.png')
        plt.close()
        print(f"Training loss curve saved to training_loss_curve.png")
    
    loss_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open("transformer_loss_history.json", "w") as f:
        json.dump(loss_data, f)
    print("Loss history saved to loss_history.json")
    
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    _, test_ppl = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Perplexity: {test_ppl:.4f}")
    
    temperature = 0.7
    
    print(f"\n--- BLEU Evaluation (Next-Token Prediction, temperature={temperature}) ---")
    bleu_score = evaluate_bleu(
        model, 
        args.test_orig_file, 
        sp, 
        device, 
        temperature=temperature, 
        eval_percentage=100
    )
    print(f"Test BLEU-1 Score (Next-Token Prediction): {bleu_score:.4f}")
    
    generate_completions(
        model, 
        args.test_orig_file, 
        sp, 
        device, 
        max_gen_length=max_seq_length, 
        temperature=temperature, 
        num_samples=5
    )
    
    print("\n--- Interactive Prompt Generation (type 'exit' to quit) ---")
    print("Type 'temp=X' to change temperature (default: 0.7)")
    print("Type 'len=X' to change maximum generation length (e.g., len=200)")
    
    current_temp = 0.7
    current_max_len = max_seq_length
    
    while True:
        user_input = input("Enter a prompt: ")
        
        if user_input.strip().lower() == "exit":
            break
        
        if user_input.startswith("temp="):
            try:
                current_temp = float(user_input[5:])
                print(f"Temperature set to {current_temp}")
            except ValueError:
                print("Invalid temperature value. Using default.")
            continue
        
        if user_input.startswith("len="):
            try:
                current_max_len = int(user_input[4:])
                print(f"Maximum generation length set to {current_max_len}")
            except ValueError:
                print("Invalid length value. Using default.")
            continue
        
        gen_text = model.prompt(
            user_input, 
            sp, 
            device, 
            max_gen_length=current_max_len, 
            temperature=current_temp
        )
        print("Generated text:")
        print(gen_text)
        print("-" * 40)
    
    print("\n--- Model Statistics ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    main()
