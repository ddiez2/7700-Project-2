import torch
import torch.nn as nn
import sys
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json

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

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_type(model):
    """Count parameters by layer type."""
    params_by_type = defaultdict(int)
    
    for name, module in model.named_modules():
        if list(module.parameters()):  # Check if module has parameters
            module_type = module.__class__.__name__
            if module_type != type(model).__name__:  # Skip the model itself
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                params_by_type[module_type] += params
                
    return params_by_type

def analyze_model_architecture(model, model_name):
    """Analyze model architecture and return details."""
    print(f"\n{'=' * 50}")
    print(f"MODEL ANALYSIS: {model_name.upper()}")
    print(f"{'=' * 50}")
    
    # Get parameter count
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Parameters by layer type
    params_by_type = count_parameters_by_type(model)
    print("\nParameters by layer type:")
    for layer_type, params in params_by_type.items():
        percentage = params / total_params * 100
        print(f"  {layer_type}: {params:,} ({percentage:.2f}%)")
    
    # Architecture details specific to each model type
    print("\nArchitecture details:")
    if model_name == "transformer":
        print(f"  Model dimension (d_model): {model.d_model}")
        print(f"  Number of attention heads: {model.transformer_encoder.layers[0].self_attn.num_heads}")
        print(f"  Number of layers: {len(model.transformer_encoder.layers)}")
        print(f"  Attention mechanism: Multi-head self-attention with causal masking")
        print(f"  Positional encoding: Sinusoidal (fixed, not learned)")
        print(f"  Weight tying: Yes (embedding weights shared with output layer)")
        print(f"  Maximum sequence length: {model.max_seq_length}")
    elif model_name == "lstm":
        print(f"  Hidden dimension: {model.d_model}")
        print(f"  Number of layers: {model.num_layers}")
        print(f"  Recurrent unit: LSTM (Long Short-Term Memory)")
        print(f"  Bidirectional: No")
        print(f"  Weight tying: Yes (embedding weights shared with output layer)")
        print(f"  Maximum sequence length: {model.max_seq_length}")
    elif model_name == "rnn":
        print(f"  Hidden dimension: {model.d_model}")
        print(f"  Number of layers: {model.num_layers}")
        print(f"  Recurrent unit: Simple RNN with tanh activation")
        print(f"  Bidirectional: No")
        print(f"  Weight tying: Yes (embedding weights shared with output layer)")
        print(f"  Maximum sequence length: {model.max_seq_length}")
    
    # Return essential data for comparison
    return {
        "name": model_name,
        "total_params": total_params,
        "params_by_type": params_by_type,
        "d_model": model.d_model,
        "num_layers": len(model.transformer_encoder.layers) if model_name == "transformer" else model.num_layers
    }

def plot_parameter_comparison(models_data):
    """Create a visual comparison of the models' parameters."""
    # Extract data
    names = [data["name"] for data in models_data]
    params = [data["total_params"] for data in models_data]
    
    # Create bar chart for total parameters
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, params, color=['#2C3E50', '#E74C3C', '#3498DB'])
    plt.title('Total Trainable Parameters by Model Architecture', fontsize=15)
    plt.ylabel('Number of Parameters', fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add parameter count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(params),
                f'{height:,}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('model_parameter_comparison.png', dpi=300, bbox_inches='tight')
    print("Bar chart saved as 'model_parameter_comparison.png'")
    
    # Create stacked bar chart for parameter distribution
    param_types = set()
    for data in models_data:
        param_types.update(data["params_by_type"].keys())
    
    param_types = sorted(list(param_types))
    data_matrix = []
    
    for data in models_data:
        model_data = []
        for param_type in param_types:
            model_data.append(data["params_by_type"].get(param_type, 0))
        data_matrix.append(model_data)
    
    # Create stacked bar chart
    plt.figure(figsize=(14, 7))
    bottom = np.zeros(len(names))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_types)))
    
    for i, param_type in enumerate(param_types):
        values = [data_matrix[j][i] for j in range(len(names))]
        plt.bar(names, values, bottom=bottom, label=param_type, color=colors[i])
        bottom += values
    
    plt.title('Parameter Distribution by Layer Type', fontsize=15)
    plt.ylabel('Number of Parameters', fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(title="Layer Types")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_parameter_distribution.png', dpi=300, bbox_inches='tight')
    print("Stacked bar chart saved as 'model_parameter_distribution.png'")

def compare_embeddings():
    """Compare the embedding processes of the three models."""
    comparison = {
        "transformer": {
            "embedding_process": "Token embeddings + sinusoidal positional encodings",
            "position_information": "Explicit through additive sinusoidal positional encodings",
            "context_mechanism": "Multi-head self-attention with causal masking",
            "parallelization": "Fully parallel (processes all tokens at once)",
            "advantages": [
                "Can capture long-range dependencies",
                "Efficient parallel processing",
                "Attention mechanism focuses on relevant context",
                "No vanishing gradient problem across sequence length"
            ],
            "disadvantages": [
                "Requires explicit positional encoding",
                "Quadratic memory complexity with sequence length",
                "More parameters than RNN/LSTM for same hidden size"
            ]
        },
        "lstm": {
            "embedding_process": "Token embeddings only (no positional encodings needed)",
            "position_information": "Implicit through sequential processing",
            "context_mechanism": "Cell state and gates (forget, input, output)",
            "parallelization": "Sequential processing only",
            "advantages": [
                "Handles long-term dependencies better than RNN",
                "Cell state prevents vanishing gradients",
                "No need for explicit position encoding",
                "Gating mechanisms control information flow"
            ],
            "disadvantages": [
                "Sequential processing limits parallelization",
                "Still can struggle with very long sequences",
                "More complex than vanilla RNN"
            ]
        },
        "rnn": {
            "embedding_process": "Token embeddings only (no positional encodings needed)",
            "position_information": "Implicit through sequential processing",
            "context_mechanism": "Simple hidden state updated at each step",
            "parallelization": "Sequential processing only",
            "advantages": [
                "Simplest architecture with fewest parameters",
                "No need for explicit position encoding",
                "Efficient for short sequences"
            ],
            "disadvantages": [
                "Suffers from vanishing/exploding gradients",
                "Poor at capturing long-range dependencies",
                "Information can be easily overwritten in hidden state",
                "Sequential processing limits parallelization"
            ]
        }
    }
    
    # Save the comparison as a JSON file
    with open('embedding_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Also create a more readable HTML file
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Embedding Processes Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
            h2 { color: #0066cc; margin-top: 30px; }
            .model { background-color: #f9f9f9; border-left: 5px solid #0066cc; padding: 15px; margin: 20px 0; }
            .transformer { border-left-color: #e74c3c; }
            .lstm { border-left-color: #27ae60; }
            .rnn { border-left-color: #3498db; }
            ul { margin-top: 10px; }
            li { margin-bottom: 5px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .advantages { color: #27ae60; }
            .disadvantages { color: #e74c3c; }
        </style>
    </head>
    <body>
        <h1>Model Embedding Processes Comparison</h1>
        
        <table>
            <tr>
                <th>Feature</th>
                <th>Transformer</th>
                <th>LSTM</th>
                <th>RNN</th>
            </tr>
            <tr>
                <td>Embedding Process</td>
                <td>Token embeddings + positional encodings</td>
                <td>Token embeddings only</td>
                <td>Token embeddings only</td>
            </tr>
            <tr>
                <td>Position Information</td>
                <td>Explicit sinusoidal encodings</td>
                <td>Implicit through sequential processing</td>
                <td>Implicit through sequential processing</td>
            </tr>
            <tr>
                <td>Context Mechanism</td>
                <td>Multi-head self-attention</td>
                <td>Cell state and gates</td>
                <td>Simple hidden state</td>
            </tr>
            <tr>
                <td>Parallelization</td>
                <td>Fully parallel</td>
                <td>Sequential only</td>
                <td>Sequential only</td>
            </tr>
        </table>

        <h2>Transformer</h2>
        <div class="model transformer">
            <p><strong>Embedding Process:</strong> The Transformer model combines token embeddings with sinusoidal positional encodings. Since Transformers process all tokens in parallel, they need explicit position information.</p>
            <p><strong>Context Handling:</strong> Uses multi-head self-attention where each token attends to all other tokens in the sequence. A causal mask ensures tokens only attend to previous tokens for language modeling.</p>
            <p><strong>Advantages:</strong></p>
            <ul class="advantages">
                <li>Excellent at capturing long-range dependencies</li>
                <li>Efficient parallel processing (faster training on modern hardware)</li>
                <li>Attention mechanism focuses on relevant parts of the context</li>
                <li>No vanishing gradient problem across sequence length</li>
            </ul>
            <p><strong>Disadvantages:</strong></p>
            <ul class="disadvantages">
                <li>Requires explicit positional encoding</li>
                <li>Quadratic memory complexity with sequence length</li>
                <li>Typically has more parameters than RNN/LSTM for the same hidden size</li>
            </ul>
        </div>

        <h2>LSTM (Long Short-Term Memory)</h2>
        <div class="model lstm">
            <p><strong>Embedding Process:</strong> LSTM uses token embeddings without needing explicit positional encodings, as position information is captured through sequential processing.</p>
            <p><strong>Context Handling:</strong> Maintains both hidden state and cell state that are passed between time steps. The cell state helps preserve information over longer sequences, while gates control information flow.</p>
            <p><strong>Advantages:</strong></p>
            <ul class="advantages">
                <li>Handles long-term dependencies much better than vanilla RNN</li>
                <li>Cell state prevents vanishing gradients</li>
                <li>No need for explicit position encoding</li>
                <li>Gating mechanisms provide fine control over information flow</li>
            </ul>
            <p><strong>Disadvantages:</strong></p>
            <ul class="disadvantages">
                <li>Sequential processing limits parallelization</li>
                <li>Can still struggle with very long sequences</li>
                <li>More complex computation than vanilla RNN</li>
            </ul>
        </div>

        <h2>RNN (Recurrent Neural Network)</h2>
        <div class="model rnn">
            <p><strong>Embedding Process:</strong> Like LSTM, vanilla RNN uses token embeddings with position information implicitly captured through sequential processing.</p>
            <p><strong>Context Handling:</strong> Uses a simple hidden state that's updated at each time step using a tanh activation function.</p>
            <p><strong>Advantages:</strong></p>
            <ul class="advantages">
                <li>Simplest architecture with fewest parameters</li>
                <li>No need for explicit position encoding</li>
                <li>Efficient for short sequences</li>
            </ul>
            <p><strong>Disadvantages:</strong></p>
            <ul class="disadvantages">
                <li>Suffers from vanishing/exploding gradients on longer sequences</li>
                <li>Poor at capturing long-range dependencies</li>
                <li>Information can be easily overwritten in hidden state</li>
                <li>Sequential processing limits parallelization</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open('embedding_comparison.html', 'w') as f:
        f.write(html_content)
    
    print("Embedding comparison saved to 'embedding_comparison.json' and 'embedding_comparison.html'")

def main():
    # Create model instances with the same architecture as in training
    vocab_size = 10000
    d_model = 128
    max_seq_length = 512
    
    transformer_model = TransformerLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        nhead=4, 
        num_layers=2,
        max_seq_length=max_seq_length, 
        dropout=0.1, 
        eos_token=3
    )
    
    lstm_model = LSTMLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_layers=2,
        dropout=0.1,
        eos_token=3,
        max_seq_length=max_seq_length
    )
    
    rnn_model = RNNLM(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_layers=2,
        dropout=0.1,
        eos_token=3,
        max_seq_length=max_seq_length
    )
    
    # Analyze each model
    transformer_data = analyze_model_architecture(transformer_model, "transformer")
    lstm_data = analyze_model_architecture(lstm_model, "lstm")
    rnn_data = analyze_model_architecture(rnn_model, "rnn")
    
    # Plot comparisons
    plot_parameter_comparison([transformer_data, lstm_data, rnn_data])
    
    # Compare embedding processes
    compare_embeddings()
    
    print("\nSuccessfully analyzed all three model architectures.")
    print("Check the generated files for detailed comparisons.")

if __name__ == "__main__":
    main()
