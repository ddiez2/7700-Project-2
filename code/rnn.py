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
import matplotlib.pyplot as plt  # For plotting loss curves

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
                # Convert tokens to integers if needed.
                if tokens and isinstance(tokens[0], str):
                    if tokens[0].isdigit():
                        tokens = list(map(int, tokens))
                    else:
                        if sp is None:
                            raise ValueError("Tokens are strings but no SentencePiece processor was provided.")
                        tokens = [sp.piece_to_id(token) for token in tokens]
                if len(tokens) < 2:
                    continue
                tokens = tokens[:max_seq_length]
                self.samples.append(tokens)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # For next-token prediction: input = tokens[:-1] and target = tokens[1:]
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
# 2. Define the RNN Language Model
# ======================================================

class RNNLM(nn.Module):
    """
    A vanilla RNN-based language model for next-token prediction.

    Architecture:
      - Embedding layer to convert token IDs into embeddings.
      - RNN layers (with dropout) for sequential modeling.
      - A fully connected output layer to project RNN outputs to vocabulary space.
    
    Weight tying is applied by sharing the embedding weights with the output layer.
    
    The forward() method returns either the logits (for training) or samples the next token
    from the final time step when sampling is enabled.
    
    The prompt() method uses autoregressive text generation.
    """
    def __init__(self, vocab_size, d_model=256, num_layers=2, dropout=0.2, eos_token=3, max_seq_length=512):
        super(RNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.eos_token = eos_token
        self.max_seq_length = max_seq_length

        # Token embeddings.
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # RNN layers.
        self.rnn = nn.RNN(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=num_layers, 
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        
        # Fully connected output layer (with weight tying).
        self.fc = nn.Linear(d_model, vocab_size)
        self.fc.weight = self.embedding.weight

    def forward(self, x, temperature=1.0, sample=False):
        """
        Forward pass through the RNN.

        Args:
            x: Tensor of shape (batch, seq_len) containing token IDs.
            temperature: Temperature scaling for sampling.
            sample: If True, sample the next token from the final time step.
        
        Returns:
            - If sample == False: logits of shape (batch, seq_len, vocab_size).
            - If sample == True: predicted next token IDs (batch,).
        """
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = self.dropout(embedded)
        # Obtain RNN outputs.
        outputs, _ = self.rnn(embedded)
        outputs = self.dropout(outputs)
        logits = self.fc(outputs)
        
        if sample:
            # Sample from the logits at the last time step.
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            return next_token.squeeze(1)
        else:
            return logits

    def prompt(self, text, sp, device, max_gen_length=None, temperature=1.0):
        """
        Autoregressively generates text given an input prompt.

        Args:
            text: Input prompt (string).
            sp: SentencePieceProcessor for tokenization/decoding.
            device: Torch device.
            max_gen_length: Maximum generation length.
            temperature: Temperature for sampling.
        
        Returns:
            Generated text (string).
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
    Only one token is predicted after each prompt for next-token prediction evaluation.
    
    Args:
        model: The RNN language model.
        test_filepath: Path to the test JSONL file.
        sp: SentencePieceProcessor.
        device: Torch device.
        max_gen_length: Maximum sequence length (unused here).
        temperature: Sampling temperature.
        eval_percentage: Percentage of test examples to evaluate.
    """
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
            predicted_token = predict_next_token(prompt_text).strip()
        except Exception as e:
            print(f"Error predicting next token: {e}")
            predicted_token = ""
        
        if len(hypotheses) < 3:
            print(f"\nSample {len(hypotheses)+1}:")
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
    
    correct = sum(1 for ref, hyp in zip(references, hypotheses) if hyp[0] == ref[0])
    accuracy = (correct / len(references)) * 100
    print(f"\nBLEU-1 Score: {bleu_1:.4f}")
    print(f"Exact match accuracy: {accuracy:.2f}%")
    print(f"Correct predictions: {correct}/{len(references)}")
    
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
        generated_completion = generated_text[len(prompt_text):].strip() if prompt_text in generated_text else generated_text.strip()
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

# ======================================================
# 4. Main Training and Evaluation Procedure
# ======================================================

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
    
    # Hyperparameters and configuration.
    vocab_size = 10000
    batch_size = 128
    lr = 2e-3 
    patience = 3
    max_seq_length = 512
    model_save_path = "best_rnn_model.pt"
    
    # Model architecture parameters.
    d_model = 128       # Embedding/hidden dimension.
    num_layers = 2      # Number of RNN layers.
    dropout = 0.1
    clip_grad_norm = 0.1
    eval_every = 1
    
    train_losses = []
    val_losses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load SentencePiece model.
    sp = spm.SentencePieceProcessor()
    sp_model_path = "../tokenizer/tokenizer.model"
    sp.load(sp_model_path)
    
    # Prepare datasets.
    train_full_dataset = TokenizedDataset(args.train_file, sp=sp, max_seq_length=max_seq_length)
    train_size = int(0.9 * len(train_full_dataset))
    val_size = len(train_full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_full_dataset, [train_size, val_size])
    test_dataset = TokenizedDataset(args.test_file, sp=sp, max_seq_length=max_seq_length)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
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
    
    # Create the RNN model.
    model = RNNLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_layers=num_layers,
        dropout=dropout,
        eos_token=3,
        max_seq_length=max_seq_length
    )
    
    # Weight initialization.
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
                
    model.apply(_initialize_weights)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=1,
        verbose=True
    )
    
    best_loss = float('inf')
    patience_counter = 0
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    total_training_start = time.time()
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    logits = model(inputs, sample=False)
                    loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs, sample=False)
                loss = criterion(logits.view(-1, model.vocab_size), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Time: {epoch_elapsed:.2f}s")
        
        if epoch % eval_every == 0:
            val_loss, val_ppl = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}")
            scheduler.step(val_loss)
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
            val_losses.append(None)
            scheduler.step(train_loss)
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
    
    # Prepare losses for plotting.
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
        print("Training loss curve saved to training_loss_curve.png")
    
    loss_data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    with open("rnn_loss_history.json", "w") as f:
        json.dump(loss_data, f)
    print("Loss history saved to rnn_loss_history.json")
    
    model.load_state_dict(torch.load(model_save_path))
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
        gen_text = model.prompt(user_input, sp, device, max_gen_length=current_max_len, temperature=current_temp)
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
