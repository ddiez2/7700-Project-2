import os
import glob
import json
import sentencepiece as spm

def train_sentencepiece(input_glob, model_prefix, vocab_size, model_type="bpe"):
    """
    Trains a SentencePiece tokenizer using all .txt files matching the input glob.
    
    Parameters:
      - input_glob: Glob pattern for your raw text files (e.g. "../data/raw/*.txt")
      - model_prefix: Full path prefix where the model and vocab will be saved.
                      For example, "../tokenizer/tokenizer" will create:
                      "../tokenizer/tokenizer.model" and "../tokenizer/tokenizer.vocab"
      - vocab_size: Desired vocabulary size (e.g. 10000)
      - model_type: Type of SentencePiece model to train ('bpe', 'unigram', etc.)
    """
    # Collect all text files.
    files = glob.glob(input_glob)
    if not files:
        raise ValueError("No text files found using the pattern: " + input_glob)
    
    # Create a comma-separated list of input files.
    input_files = ",".join(files)
    print(f"Found {len(files)} files for training.")

    # Train the SentencePiece model.
    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,  # Adjust if working with non-English text.
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print(f"Tokenizer model and vocab saved as {model_prefix}.model and {model_prefix}.vocab")

def tokenize_jsonl(input_file, output_file, sp):
    """
    Tokenizes each JSON object in a JSONL file by combining the "prompt" and "completion" fields.
    
    The function expects each JSON object to have at least the following keys:
      - "prompt"
      - "completion"
    
    The two fields are concatenated (with a space between them). The resulting text is
    tokenized, and the output JSON (written to output_file) contains only the tokenized output
    under the key "tokens".
    
    This way the model will see only the tokenized text, i.e. the prompt as context and the tokens
    for the completion as the continuation. You can later decide in your training code to compute loss
    only for the completion tokens.
    """
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())
            if "prompt" in data and "completion" in data:
                # Combine prompt and completion with a space in between.
                combined_text = data["prompt"].strip() + " " + data["completion"].strip()
                # Tokenize the combined text. (Use out_type=str for subword pieces; use int for IDs.)
                tokens = sp.encode(combined_text, out_type=str)
                # Store only the tokenized sequence.
                new_data = {"tokens": tokens}
            else:
                raise KeyError(
                    "Expected keys 'prompt' and 'completion' in JSON object, but got: " + str(data.keys())
                )
            fout.write(json.dumps(new_data) + "\n")
    print(f"Tokenized data saved to {output_file}")

if __name__ == "__main__":
    # Define paths and parameters.
    raw_data_pattern = "../data/raw/*.txt"
    # Save the model and vocab to the ../tokenizer directory.
    model_prefix = "../tokenizer/tokenizer"
    vocab_size = 10000

    # Step 1: Train the SentencePiece model.
    train_sentencepiece(raw_data_pattern, model_prefix, vocab_size)
    
    # Step 2: Load the trained SentencePiece model.
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    print("Loaded SentencePiece model.")
    
    # Step 3: Tokenize the JSONL files.
    data_dir = "../data"
    for split in ["train", "test"]:
        input_jsonl = os.path.join(data_dir, f"{split}.jsonl")
        output_jsonl = os.path.join(data_dir, f"{split}_tokenized.jsonl")
        tokenize_jsonl(input_jsonl, output_jsonl, sp)
    
    print("Tokenization complete for both train and test datasets.")
