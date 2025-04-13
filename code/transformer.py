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
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# ======================================================
# 1. Dataset and Collate Functions for Tokenized Files
# ======================================================

class TokenizedDataset(Dataset):
    """
    Loads tokenized JSONL files where each JSON object contains a "tokens"
    field (a list of subword tokens or token IDs).
    
    For teacher forcing, each sample uses tokens[:-1] as input and tokens[1:] as target.
    If tokens are stored as strings that are not numeric, a SentencePiece processor (sp)
    is used to convert them to integer IDs.
    """
    def __init__(self, filepath, sp=None, max_seq_length=512):
        self.samples = []
        self.max_seq_length = max_seq_length
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                tokens = data["tokens"]
                # Check if tokens are strings.
                if tokens and isinstance(tokens[0], str):
                    if tokens[0].isdigit():
                        tokens = list(map(int, tokens))
                    else:
                        if sp is None:
                            raise ValueError("Tokens are of string type but no SentencePiece processor was provided for conversion.")
                        tokens = [sp.piece_to_id(token) for token in tokens]
                # Only keep sequences with at least two tokens.
                if len(tokens) < 2:
                    continue
                tokens = tokens[:max_seq_length]
                self.samples.append(tokens)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # For next-token prediction: input is tokens[:-1], target is tokens[1:]
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def collate_fn(batch, pad_token=0):
    """
    Pads a batch of (input, target) pairs so that all sequences in the batch have the same length.
    """
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_len = max(seq.size(0) for seq in inputs)
    padded_inputs = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), value=pad_token) for seq in inputs])
    padded_targets = torch.stack([F.pad(seq, (0, max_len - seq.size(0)), value=pad_token) for seq in targets])
    return padded_inputs, padded_targets

# ======================================================
# 2. Define the Transformer Language Model
# ======================================================

class TransformerLM(nn.Module):
    """
    A small transformer language model for next-token prediction.
    
    Architecture:
      - Embedding layer for mapping token IDs to embeddings.
      - Sinusoidal positional encodings (fixed, not learned).
      - Transformer encoder layers (with multi-head attention).
      - Fully connected output layer for predicting vocabulary probabilities.
    
    The forward() method can either return logits (for training) or sample the next token
    based on a given temperature (for text generation). The prompt() method autoregressively
    generates a continuation given an input prompt.
    
    Note: Assumes the EOS token ID is 3.
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4,
                 max_seq_length=512, dropout=0.2, eos_token=3):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.eos_token = eos_token
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Create sinusoidal positional encodings (fixed, not learned)
        self.register_buffer("positional_encoding", self.create_sinusoidal_encoding(max_seq_length, d_model))
        self.pos_dropout = nn.Dropout(p=dropout)
        
        # Transformer encoder layers - using batch_first=True to avoid nested tensor warning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected output layer - using weight tying for better generalization
        self.fc = nn.Linear(d_model, vocab_size)
        # Weight tying: share weights between embedding and output layer
        self.fc.weight = self.embedding.weight
    
    def create_sinusoidal_encoding(self, max_len, d_model):
        """
        Create sinusoidal position encodings.
        
        :param max_len: Maximum sequence length
        :param d_model: Dimension of the model
        :return: Tensor of shape (1, max_len, d_model)
        """
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x, temperature=1.0, sample=False):
        """
        Forward pass.
        
        :param x: Tensor of shape (batch, seq_len) with token IDs.
        :param temperature: Sampling temperature.
        :param sample: If True, sample the next token; otherwise, return logits.
        :returns: Logits tensor (if sample is False) or next token IDs (if sample is True).
        """
        batch_size, seq_len = x.size()
        
        # Get token embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)  # Scaling by sqrt(d_model) improves training
        
        # Add positional encodings - just use the first seq_len positions
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.pos_dropout(x)
        
        # Create causal mask - shape needs to match batch_first=True format
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # No need to transpose since we're using batch_first=True
        x = self.transformer_encoder(x, mask)
        logits = self.fc(x)    # (batch, seq_len, vocab_size)
        
        if sample:
            # Sample next token from last time step using temperature scaling.
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            return next_token.squeeze(1)
        else:
            return logits
    
    def prompt(self, text, sp, device, max_gen_length=None, temperature=1.0):
        """
        Autoregressively generates text given an input prompt.
        
        :param text: Input prompt (string).
        :param sp: SentencePieceProcessor instance for tokenization and decoding.
        :param device: Torch device.
        :param max_gen_length: Maximum generation length.
        :param temperature: Temperature for sampling.
        :returns: Generated text (string).
        """
        if max_gen_length is None:
            max_gen_length = self.max_seq_length
        token_ids = sp.encode(text, out_type=int)
        generated = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        self.eval()
        with torch.no_grad():
            for _ in range(max_gen_length - generated.size(1)):
                next_token = self.forward(generated, temperature=temperature, sample=True)
                # If we generated an EOS token, stop generation
                if next_token.item() == self.eos_token:
                    break
                next_token = next_token.unsqueeze(0)
                generated = torch.cat([generated, next_token], dim=1)
        
        # Get token IDs and truncate at the first EOS token if present
        generated_ids = generated.squeeze(0).tolist()
        if self.eos_token in generated_ids:
            # Find the position of the first EOS token
            eos_pos = generated_ids.index(self.eos_token)
            # Truncate the sequence up to (but not including) the EOS token
            generated_ids = generated_ids[:eos_pos]
        
        return sp.decode_ids(generated_ids)

# ======================================================
# 3. Training, Perplexity, and BLEU Evaluation Functions
# ======================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            # Updated to use torch.amp.autocast instead of torch.cuda.amp.autocast
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
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs, sample=False)
        loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
        total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def evaluate_bleu(model, test_filepath, sp, device, max_gen_length=None, temperature=1.0, eval_percentage=100):
    """
    Computes the corpus-level BLEU score by comparing the model's next-token prediction to the reference.
    Only predicts a single token after each prompt, as this is a next-token prediction task.
    
    Args:
        model: The transformer model
        test_filepath: Path to the test JSONL file
        sp: SentencePiece processor
        device: Device to run inference on
        max_gen_length: Maximum sequence length (not used in single-token prediction)
        temperature: Temperature for sampling
        eval_percentage: Percentage of test data to use
    """
    references = []
    hypotheses = []
    
    # Load all lines from test file
    with open(test_filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    # If not using full test set, sample a subset
    test_size = len(all_lines)
    if eval_percentage < 100:
        # Calculate how many examples to use
        sample_size = max(1, int(test_size * eval_percentage / 100))
        # Use a fixed seed for reproducibility
        import random
        random.seed(42)
        # Sample without replacement
        sampled_lines = random.sample(all_lines, sample_size)
        print(f"BLEU evaluation on {eval_percentage}% of test data ({sample_size}/{test_size} examples)")
    else:
        sampled_lines = all_lines
        print(f"BLEU evaluation on 100% of test data ({test_size} examples)")
    
    # Define a function to predict only the next token
    def predict_next_token(text):
        # Encode the prompt
        token_ids = sp.encode(text, out_type=int)
        input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Get the prediction for the next token
            next_token_id = model.forward(input_tensor, temperature=temperature, sample=True).item()
            
            # Decode just the predicted token
            next_token_text = sp.decode_ids([next_token_id])
            return next_token_text
    
    # Process the selected lines
    for line in tqdm(sampled_lines, desc="BLEU Evaluating", leave=False):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]
        
        # Get first word/token of the reference (what we're trying to predict)
        reference_first_token = reference_text.split()[0] if reference_text else ""
        
        # Predict just the next token after the prompt
        try:
            predicted_token = predict_next_token(prompt_text)
            predicted_token = predicted_token.strip()
        except Exception as e:
            print(f"Error predicting next token: {e}")
            predicted_token = ""
        
        # Print sample predictions for the first few examples
        if len(hypotheses) < 3:
            print(f"\nSample {len(hypotheses) + 1}:")
            print(f"Prompt: {prompt_text}")
            print(f"Reference token: '{reference_first_token}'")
            print(f"Predicted token: '{predicted_token}'")
            print("-" * 40)
        
        # Add to references and hypotheses for BLEU calculation
        references.append([reference_first_token])
        hypotheses.append([predicted_token])
    
    # Use smoothing function for better BLEU scores with short sequences
    smoothing = SmoothingFunction().method1
    
    # Calculate BLEU scores for different n-gram orders (though mostly BLEU-1 is relevant here)
    # Only calculate BLEU-1 since we're predicting single tokens
    bleu_1 = corpus_bleu(
        references, 
        hypotheses, 
        weights=(1, 0, 0, 0), 
        smoothing_function=smoothing
    )
    
    # Also calculate simple accuracy (exact match)
    correct = 0
    total = len(references)
    for ref, hyp in zip(references, hypotheses):
        if hyp[0] == ref[0]:
            correct += 1
    accuracy = (correct / total) * 100
    
    # Print results
    print(f"\nBLEU-1 Score: {bleu_1:.4f}")
    print(f"Exact match accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{total}")
    
    # Return the BLEU-1 score
    return bleu_1

def generate_completions(model, test_filepath, sp, device, max_gen_length=None, temperature=1.0, num_samples=5):
    """
    Generates full text completions for prompts in the test file.
    This is separate from the BLEU evaluation and is used to demonstrate the model's 
    text generation capabilities.
    
    Args:
        model: The transformer model
        test_filepath: Path to the test JSONL file
        sp: SentencePiece processor
        device: Device to run inference on
        max_gen_length: Maximum generation length
        temperature: Temperature for generation
        num_samples: Number of samples to generate
    """
    # Load test file
    with open(test_filepath, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    # Select a few samples
    import random
    random.seed(42)  # For reproducibility
    sample_lines = random.sample(all_lines, min(num_samples, len(all_lines)))
    
    print(f"\n--- Full Text Generation Examples (temperature={temperature}) ---")
    
    # Generate completions for each sample
    for i, line in enumerate(sample_lines):
        data = json.loads(line.strip())
        prompt_text = data["prompt"]
        reference_text = data["completion"]
        
        # Generate full completion
        generated_text = model.prompt(prompt_text, sp, device, max_gen_length=max_gen_length, temperature=temperature)
        
        # Extract just the completion part
        if prompt_text in generated_text:
            generated_completion = generated_text[len(prompt_text):].strip()
        else:
            generated_completion = generated_text.strip()
        
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompt_text}")
        print(f"Reference: {reference_text}")
        print(f"Generated: {generated_completion}")
        print("-" * 40)

# ======================================================
# 4. Main Training and Evaluation Procedure
# ======================================================

# Add this function to plot loss curves
def plot_loss_curves(train_losses, val_losses, save_path="transformer_loss_curve.png"):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
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
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    
    # Hardcode the hyperparameters instead of using CLI arguments
    vocab_size = 10000
    batch_size = 128
    lr = 2e-3 
    patience = 3
    max_seq_length = 512
    model_save_path = "best_transformer_model.pt"
    
    # Model architecture parameters - REDUCED SIZE for quicker training
    d_model = 128  # Reduced from 256
    nhead = 4      # Reduced from 8
    num_layers = 2  # Reduced from 4
    dropout = 0.1  # Lower dropout for initial training
    clip_grad_norm = 0.1  
    eval_every = 1
    
    # Lists to track losses for plotting
    train_losses = []
    val_losses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the SentencePiece model.
    sp = spm.SentencePieceProcessor()
    sp_model_path = "../tokenizer/tokenizer.model"
    sp.load(sp_model_path)
    
    # Load tokenized datasets and split train into train and validation sets
    train_full_dataset = TokenizedDataset(args.train_file, sp=sp, max_seq_length=max_seq_length)
    
    # Split training data into train and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, val_size])
    
    # Keep test dataset separate for final evaluation only
    test_dataset = TokenizedDataset(args.test_file, sp=sp, max_seq_length=max_seq_length)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # DataLoader optimizations.
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
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Build the transformer model with improved architecture
    model = TransformerLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        nhead=nhead, 
        num_layers=num_layers,
        max_seq_length=max_seq_length, 
        dropout=dropout, 
        eos_token=3
    )
    
    # Initialize weights properly
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    model.apply(_initialize_weights)
    model.to(device)
    
    # Use label smoothing for better convergence
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # Use simpler Adam optimizer with smaller learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    
    # Use simpler scheduler for more predictable learning rate decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True
    )
    
    best_loss = float('inf')
    patience_counter = 0

    # Fix GradScaler initialization 
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    total_training_start = time.time()
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    logits = model(inputs, sample=False)
                    loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                scaler.scale(loss).backward()
                # Add gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs, sample=False)
                loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                loss.backward()
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Update progress bar with current learning rate
            progress_bar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)  # Save training loss for plotting
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Time: {epoch_elapsed:.2f}s")
        
        # Periodically evaluate on validation set
        if epoch % eval_every == 0:
            val_loss, val_ppl = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)  # Save validation loss for plotting
            print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}")
            
            # Use scheduler to adjust learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Save model if validation loss improves
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                print("Model improved; saving model.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            # If not evaluating this epoch, use training loss for early stopping
            val_losses.append(None)  # Add None for epochs without validation
            scheduler.step(train_loss)  # Use training loss for scheduler
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
                print("Model improved; saving model.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
    
    total_training_elapsed = time.time() - total_training_start
    print(f"Total training time: {total_training_elapsed:.2f}s")
    
    # Clean up the validation losses list (remove None values)
    val_losses_cleaned = []
    train_losses_for_plot = []
    
    for i, val_loss in enumerate(val_losses):
        if val_loss is not None:
            val_losses_cleaned.append(val_loss)
            train_losses_for_plot.append(train_losses[i])
    
    # Plot and save the loss curves
    if len(val_losses_cleaned) > 0:  # Only plot if we have validation data
        plot_loss_curves(train_losses_for_plot, val_losses_cleaned)
    else:  # Otherwise just plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('training_loss_curve.png')
        plt.close()
        print(f"Training loss curve saved to training_loss_curve.png")
    
    # Save the loss values to a JSON file for future reference
    loss_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open("transformer_loss_history.json", "w") as f:
        json.dump(loss_data, f)
    print("Loss history saved to loss_history.json")
    
    # Load the best saved model.
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)
    
    # Only use test set for final evaluation after training is complete
    _, test_ppl = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Perplexity: {test_ppl:.4f}")
    
    # Use a single temperature of 0.7 for BLEU evaluation
    temperature = 0.7
    
    # Evaluate BLEU for next-token prediction
    print(f"\n--- BLEU Evaluation (Next-Token Prediction, temperature={temperature}) ---")
    bleu_score = evaluate_bleu(
        model, 
        args.test_orig_file, 
        sp, 
        device, 
        temperature=temperature,
        eval_percentage=100  # Use full dataset for final evaluation
    )
    print(f"Test BLEU-1 Score (Next-Token Prediction): {bleu_score:.4f}")
    
    # Also demonstrate full text generation on a few examples
    generate_completions(
        model,
        args.test_orig_file,
        sp,
        device,
        max_gen_length=max_seq_length,
        temperature=temperature,
        num_samples=5
    )
    
    # Interactive prompt generation with default temperature of 0.7
    print("\n--- Interactive Prompt Generation (type 'exit' to quit) ---")
    print("Type 'temp=X' to change temperature (default: 0.7)")
    print("Type 'len=X' to change maximum generation length (e.g., len=200)")
    
    current_temp = 0.7  # Set default temperature to 0.7
    current_max_len = max_seq_length
    
    while True:
        user_input = input("Enter a prompt: ")
        
        if user_input.strip().lower() == "exit":
            break
        
        # Check for temperature change command
        if user_input.startswith("temp="):
            try:
                current_temp = float(user_input[5:])
                print(f"Temperature set to {current_temp}")
            except ValueError:
                print("Invalid temperature value. Using default.")
            continue
            
        # Check for max length change command
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

    # Add code to display model statistics
    print("\n--- Model Statistics ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")

if __name__ == "__main__":
    main()
