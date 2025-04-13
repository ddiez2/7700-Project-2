import os
import torch
import torch.nn as nn
from torchviz import make_dot
from torch.autograd import Variable
import sys

# Add the current directory to the path to import the models
sys.path.append(os.path.dirname(__file__))

# Import the model classes from their respective files
try:
    from transformer import TransformerLM
    from lstm import LSTMLM
    from rnn import RNNLM
except ImportError as e:
    print(f"Error importing model classes: {e}")
    print("Make sure the model files are in the correct directory and named correctly.")
    sys.exit(1)

def create_model_diagram(model, model_name, input_size=64, vocab_size=10000):
    """
    Creates and saves a diagram of the model architecture.
    
    Args:
        model: The PyTorch model
        model_name: Name for the output file
        input_size: Size of the dummy input
        vocab_size: Size of the vocabulary
    """
    print(f"Creating diagram for {model_name}...")
    
    # Create a dummy input
    x = Variable(torch.randint(0, vocab_size, (1, input_size)), requires_grad=False)
    
    # Forward pass to create the computational graph
    try:
        y = model(x)
        
        # Create the dot graph
        dot = make_dot(y, params=dict(list(model.named_parameters())))
        
        # Set graph attributes
        dot.attr('node', fontsize='12')
        dot.attr(rankdir='TB')  # Top to Bottom layout
        dot.attr('graph', label=f'{model_name} Architecture')
        
        # Save the graph
        output_file = f"{model_name}_diagram"
        dot.render(output_file, format='png', cleanup=True)
        print(f"Diagram saved to {output_file}.png")
        
    except Exception as e:
        print(f"Error creating diagram for {model_name}: {e}")

def main():
    # Define the model parameters (must match the parameters used during training)
    vocab_size = 10000
    d_model = 128
    max_seq_length = 512
    
    # Paths to the saved model files
    transformer_path = "./best_transformer_model.pt"
    lstm_path = "./best_lstm_model.pt"
    rnn_path = "./best_rnn_model.pt"
    
    device = torch.device("cpu")  # Use CPU for visualization
    
    # Create and visualize Transformer model
    print("\nProcessing Transformer model...")
    try:
        transformer_model = TransformerLM(
            vocab_size=vocab_size, 
            d_model=d_model, 
            nhead=4, 
            num_layers=2,
            max_seq_length=max_seq_length, 
            dropout=0.1, 
            eos_token=3
        )
        transformer_model.load_state_dict(torch.load(transformer_path, map_location=device))
        transformer_model.eval()
        create_model_diagram(transformer_model, "transformer", input_size=32)
    except Exception as e:
        print(f"Error loading Transformer model: {e}")
    
    # Create and visualize LSTM model
    print("\nProcessing LSTM model...")
    try:
        lstm_model = LSTMLM(
            vocab_size=vocab_size, 
            d_model=d_model, 
            num_layers=2,
            dropout=0.1,
            eos_token=3,
            max_seq_length=max_seq_length
        )
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm_model.eval()
        create_model_diagram(lstm_model, "lstm", input_size=32)
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
    
    # Create and visualize RNN model
    print("\nProcessing RNN model...")
    try:
        rnn_model = RNNLM(
            vocab_size=vocab_size, 
            d_model=d_model, 
            num_layers=2,
            dropout=0.1,
            eos_token=3,
            max_seq_length=max_seq_length
        )
        rnn_model.load_state_dict(torch.load(rnn_path, map_location=device))
        rnn_model.eval()
        create_model_diagram(rnn_model, "rnn", input_size=32)
    except Exception as e:
        print(f"Error loading RNN model: {e}")
    
    print("\nAll diagrams have been created.")

if __name__ == "__main__":
    main()
