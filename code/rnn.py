import os 
import json
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt

# Dataset for tokenized files
class TokenizedDataset(Dataset):
    """
    Loads tokenized data from JSONL files
    """
    def __init__(self, filepath, sp=None, max_seq_length=512):
        self.samples = []
        self.max_seq_length = max_seq_length
        f = open(filepath, "r", encoding="utf-8")  # Open file directly
        for line in f:
            data = json.loads(line.strip())
            tokens = data["tokens"]
            
            # Convert string tokens to IDs
            if tokens and type(tokens[0]) == str:
                if tokens[0].isdigit():
                    tokens = [int(t) for t in tokens]
                else:
                    tokens = [sp.piece_to_id(t) for t in tokens]
            
            # Skip short sequences
            if len(tokens) <= 1:
                continue
                
            # Cut if too long
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                
            self.samples.append(tokens)
        f.close()
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        toks = self.samples[idx]
        # Input is all tokens except last, target is all tokens except first
        inputs = toks[:-1]
        targets = toks[1:]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def collate_fn(batch, pad_token=0):
    """
    Pads sequences to the same length
    """
    # Extract inputs and targets
    inputs = []
    targets = []
    for i in range(len(batch)):
        inputs.append(batch[i][0])
        targets.append(batch[i][1])
    
    # Find max length in this batch
    max_len = 0
    for seq in inputs:
        if seq.size(0) > max_len:
            max_len = seq.size(0)
    
    # Pad everything to max_len
    padded_inputs = []
    padded_targets = []
    for i in range(len(inputs)):
        # Manual padding instead of using F.pad
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
    
    # Stack them into tensors
    return torch.stack(padded_inputs), torch.stack(padded_targets)

# Enhanced RNN Language Model
class RNNLM(nn.Module):
    """
    Enhanced RNN language model with architecture comparable to Transformer and LSTM
    """
    def __init__(self, vocab_size, d_model=512, num_layers=6, dropout=0.1, eos_token=3, max_seq_length=512):
        super(RNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.eos_token = eos_token
        self.max_seq_length = max_seq_length

        # Token embeddings with scaling
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Dropout layers
        self.input_dropout = nn.Dropout(p=dropout)
        self.output_dropout = nn.Dropout(p=dropout)
        
        # Layer norm for pre and post processing (like Transformer)
        self.prenorm = nn.LayerNorm(d_model)
        self.postnorm = nn.LayerNorm(d_model)
        
        # Main RNN layers
        self.rnn = nn.RNN(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=num_layers, 
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Projection layer
        self.fc = nn.Linear(d_model, vocab_size)
        
        # Weight tying between embedding and output layer (like Transformer)
        self.fc.weight = self.embedding.weight

    def forward(self, x, temperature=1.0, sample=False):
        """
        Forward pass with transformer-like preprocessing
        """
        # Embed and scale like transformer
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        
        # Apply dropout
        embedded = self.input_dropout(embedded)
        
        # Apply prenorm (like Transformer)
        embedded = self.prenorm(embedded)
        
        # Process through RNN
        outputs, _ = self.rnn(embedded)
        
        # Apply postnorm and dropout (like Transformer)
        outputs = self.postnorm(outputs)
        outputs = self.output_dropout(outputs)
        
        # Project to vocabulary size
        logits = self.fc(outputs)
        
        if sample:
            # Sample next token (same as Transformer)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token.squeeze(1)
        else:
            return logits

    def prompt(self, text, sp, device, max_gen_length=None, temperature=1.0):
        """
        Generate text from a prompt
        """
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
        
        # Get tokens and remove EOS if present
        generated_ids = generated.squeeze(0).tolist()
        if self.eos_token in generated_ids:
            eos_pos = generated_ids.index(self.eos_token)
            generated_ids = generated_ids[:eos_pos]
        
        return sp.decode_ids(generated_ids)

# Training and evaluation functions
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_tokens = 0  # Track total non-padding tokens
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Count non-padding tokens for loss normalization
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
    total_tokens = 0  # Track total non-padding tokens
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Count non-padding tokens
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
    """
    Calculate BLEU score for next-token prediction
    """
    references = []
    hypotheses = []
    
    # Load test data
    f = open(test_filepath, "r", encoding="utf-8")
    all_lines = f.readlines()
    f.close()
    
    # Sample subset if needed
    test_size = len(all_lines)
    if eval_percentage < 100:
        sample_size = max(1, int(test_size * eval_percentage / 100))
        import random
        random.seed(42)
        sampled_lines = random.sample(all_lines, sample_size)
        print(f"BLEU evaluation on {sample_size}/{test_size} examples")
    else:
        sampled_lines = all_lines
        print(f"BLEU evaluation on all {test_size} examples")
    
    # Function to predict next token
    def predict_next_token(text):
        token_ids = sp.encode(text, out_type=int)
        input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            next_token_id = model.forward(input_tensor, temperature=temperature, sample=True).item()
            next_token_text = sp.decode_ids([next_token_id])
            return next_token_text
    
    # Process each example
    for line in tqdm(sampled_lines, desc="BLEU Evaluating", leave=False):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]
        reference_first_token = reference_text.split()[0] if reference_text else ""
        
        try:
            predicted_token = predict_next_token(prompt_text).strip()
        except:
            predicted_token = ""
        
        # Print some samples
        if len(hypotheses) < 3:
            print(f"\nSample {len(hypotheses)+1}:")
            print(f"Prompt: {prompt_text}")
            print(f"Reference: '{reference_first_token}'")
            print(f"Predicted: '{predicted_token}'")
            print("-" * 40)
        
        references.append([reference_first_token])
        hypotheses.append([predicted_token])
    
    # Calculate BLEU score
    smoothing = SmoothingFunction().method1
    bleu_1 = corpus_bleu(
        references, 
        hypotheses, 
        weights=(1, 0, 0, 0),
        smoothing_function=smoothing
    )
    
    # Calculate accuracy
    correct = 0
    for i in range(len(references)):
        if hypotheses[i][0] == references[i][0]:
            correct += 1
    accuracy = (correct / len(references)) * 100
    
    print(f"\nBLEU-1 Score: {bleu_1:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{len(references)}")
    
    return bleu_1

def generate_completions(model, test_filepath, sp, device, max_gen_length=None, temperature=1.0, num_samples=5):
    """
    Generate text completions for examples from test file
    """
    f = open(test_filepath, "r", encoding="utf-8")
    all_lines = f.readlines()
    f.close()
    
    import random
    random.seed(42)
    sample_lines = random.sample(all_lines, min(num_samples, len(all_lines)))
    
    print(f"\n--- Text Generation Examples (temp={temperature}) ---")
    for i, line in enumerate(sample_lines):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]
        
        # Generate text
        generated_text = model.prompt(
            prompt_text, 
            sp, 
            device, 
            max_gen_length=max_gen_length, 
            temperature=temperature
        )
        
        # Get just the completion part
        if prompt_text in generated_text:
            generated_completion = generated_text[len(prompt_text):].strip()
        else:
            generated_completion = generated_text.strip()
        
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompt_text}")
        print(f"Reference: {reference_text}")
        print(f"Generated: {generated_completion}")
        print("-" * 40)

def plot_loss_curves(train_losses, val_losses, save_path="rnn_loss_curve.png"):
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
    parser.add_argument("--train_file", type=str, default="../data/train_tokenized.jsonl")
    parser.add_argument("--test_file", type=str, default="../data/test_tokenized.jsonl")
    parser.add_argument("--test_orig_file", type=str, default="../data/test.jsonl")
    parser.add_argument("--epochs", type=int, default=30)  # Increased to match Transformer
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Learning rate warmup steps")
    args = parser.parse_args()
    
    # Model settings
    vocab_size = 10000  # Using same vocab size as other models for fair comparison
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    lr = args.lr
    patience = 5  # Using lower patience since RNNs can be more volatile in training
    max_seq_length = 512  # Standard sequence length for all models
    model_save_path = "best_rnn_model.pt"
    
    # RNN specific architecture decisions
    d_model = 512  # Same embedding size as Transformer/LSTM for fair comparison
    num_layers = 6  # Deep RNN to capture more complex patterns
    dropout = 0.1  # Moderate dropout to prevent vanilla RNN overfitting
    weight_decay = 0.01  # L2 regularization to improve generalization
    clip_grad_norm = 3.0  # Essential to prevent exploding gradients in RNNs
    eval_every = 1  # Evaluate more frequently due to RNN's training dynamics
    
    # Track losses
    train_losses = []
    val_losses = []
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("../tokenizer/tokenizer.model")
    
    # Load datasets
    train_full_dataset = TokenizedDataset(args.train_file, sp=sp, max_seq_length=max_seq_length)
    
    # Split into train and validation
    train_size = int(0.9 * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, val_size])
    
    # Test dataset
    test_dataset = TokenizedDataset(args.test_file, sp=sp, max_seq_length=max_seq_length)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = RNNLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_layers=num_layers,
        dropout=dropout,
        eos_token=3,
        max_seq_length=max_seq_length
    )
    
    # Initialize weights - improved initialization like Transformer
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            # Use Kaiming initialization for linear layers
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Standard layer norm initialization
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
                
    model.apply(_initialize_weights)
    model.to(device)
    
    # Loss and optimizer - updated to match Transformer
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        betas=(0.9, 0.98), 
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    # Create warmup scheduler to match Transformer
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    
    def lr_lambda(current_step: int):
        # Linear warmup followed by cosine decay
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        # Cosine annealing
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    # Training loop - updated to match Transformer with gradient accumulation
    total_training_start = time.time()
    print("Starting training...")
    
    # Track step for scheduler
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()  # Zero gradients once at the start of epoch
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    logits = model(inputs, sample=False)
                    loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                    # Normalize loss for gradient accumulation
                    loss = loss / args.accumulation_steps
                
                # Accumulate gradients
                scaler.scale(loss).backward()
                
                # Only step and zero grad after accumulation steps
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    # Unscale before clip to use correct gradient values
                    scaler.unscale_(optimizer)
                    # Add gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update scheduler
                    scheduler.step()
                    global_step += 1
            else:
                logits = model(inputs, sample=False)
                loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                # Normalize loss for gradient accumulation
                loss = loss / args.accumulation_steps
                
                # Accumulate gradients
                loss.backward()
                
                # Only step and zero grad after accumulation steps
                if (batch_idx + 1) % args.accumulation_steps == 0:
                    # Add gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update scheduler
                    scheduler.step()
                    global_step += 1
            
            # Multiply by accumulation steps to get correct loss
            train_loss += (loss.item() * args.accumulation_steps * inputs.size(0))
            
            # Update progress bar with current learning rate
            progress_bar.set_postfix({"loss": loss.item() * args.accumulation_steps, 
                                     "lr": optimizer.param_groups[0]['lr']})
        
        # Handle final accumulated batch if needed
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
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Time: {epoch_elapsed:.2f}s")
        
        # Periodically evaluate on validation set
        if epoch % eval_every == 0:
            val_loss, val_ppl = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}")
            
            # Save model if validation loss improves
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
            # If not evaluating this epoch, use training loss for early stopping
            val_losses.append(None)
    
    total_training_elapsed = time.time() - total_training_start
    print(f"Total training time: {total_training_elapsed:.2f}s")
    
    # Prepare data for plotting
    val_losses_cleaned = []
    train_losses_for_plot = []
    for i, val_loss in enumerate(val_losses):
        if (val_loss is not None):
            val_losses_cleaned.append(val_loss)
            train_losses_for_plot.append(train_losses[i])
    
    # Plot loss curves
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
        print("Training loss curve saved to training_loss_curve.png")
    
    # Save loss history
    loss_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open("rnn_loss_history.json", "w") as f:
        json.dump(loss_data, f)
    print("Loss history saved to rnn_loss_history.json")
    
    # Load best model for evaluation
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Final evaluation
    _, test_ppl = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Perplexity: {test_ppl:.4f}")
    
    # BLEU evaluation and text generation
    temperature = 0.7
    print(f"\n--- BLEU Evaluation (temperature={temperature}) ---")
    bleu_score = evaluate_bleu(
        model, 
        args.test_orig_file, 
        sp, 
        device, 
        temperature=temperature,
        eval_percentage=100
    )
    print(f"BLEU-1 Score: {bleu_score:.4f}")
    
    # Generate some examples
    generate_completions(
        model,
        args.test_orig_file,
        sp,
        device,
        max_gen_length=max_seq_length,
        temperature=temperature,
        num_samples=5
    )
    
    # Interactive mode
    print("\n--- Interactive Generation (type 'exit' to quit) ---")
    print("Type 'temp=X' to change temperature (default: 0.7)")
    print("Type 'len=X' to change max length (default: 512)")
    
    current_temp = 0.7
    current_max_len = max_seq_length
    
    while True:
        user_input = input("Enter prompt: ")
        if user_input.strip().lower() == "exit":
            break
        if user_input.startswith("temp="):
            try:
                current_temp = float(user_input[5:])
                print(f"Temperature: {current_temp}")
            except:
                print("Invalid temperature")
            continue
        if user_input.startswith("len="):
            try:
                current_max_len = int(user_input[4:])
                print(f"Max length: {current_max_len}")
            except:
                print("Invalid length")
            continue
            
        gen_text = model.prompt(
            user_input, 
            sp, 
            device, 
            max_gen_length=current_max_len, 
            temperature=current_temp
        )
        print("Generated:")
        print(gen_text)
        print("-" * 40)
    
    # Show model stats
    print("\n--- Model Stats ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    main()
