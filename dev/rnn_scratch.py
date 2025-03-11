import numpy as np
import torch
from torch._prims_common import Tensor
import torch.nn as nn
import os


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_tensor_as_bin(filename, tensor):
    """Save tensor to binary file with proper format for C consumption"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save as float32 (C's float) for most tensors
    if tensor.dtype in [torch.float32, torch.float]:
        tensor.detach().cpu().numpy().astype(np.float32).tofile(filename)
    # Save as int32 for integer tensors
    elif tensor.dtype in [torch.int32, torch.int64, torch.long, torch.int]:
        tensor.detach().cpu().numpy().astype(np.int32).tofile(filename)
    else:
        print(f"Warning: Unhandled dtype {tensor.dtype} for {filename}")
        tensor.detach().cpu().numpy().tofile(filename)

    # Also save shape information for easier loading in C
    with open(f"{filename}.shape", "w") as f:
        f.write(",".join(str(dim) for dim in tensor.shape))


def rnn_custom(x: Tensor, rnn_model: nn.RNN):
    '''
    RNN forward implementation using weights from nn.RNN
    args:
        x: torch.tensor (batch_size, seq_len, feature)
        rnn_model: nn.RNN model to get weights from
    return:
        output: torch.tensor (batch_size, seq_len, hidden_size)
        h_t: torch.tensor(num_layers, batch_size, hidden_size)
    '''
    # Get model dimensions and convert if necessary
    batch_size, seq_len, input_size = x.shape
    hidden_size = rnn_model.hidden_size
    num_layers = rnn_model.num_layers

    # Initialize hidden state with correct dimensions
    h_t = torch.zeros(num_layers, batch_size, hidden_size)

    # Prepare for storing all outputs
    outputs = torch.zeros(batch_size, seq_len, hidden_size)

    # Process each time step
    for t in range(num_layers):
        layer_input = x[:, t, :]

        # for layer in range(num_layers):
        layer = 0
        # Get layer-specific weights
        w_ih = getattr(rnn_model, f'weight_ih_l{layer}')
        w_hh = getattr(rnn_model, f'weight_hh_l{layer}')
        b_ih = getattr(rnn_model, f'bias_ih_l{layer}')
        b_hh = getattr(rnn_model, f'bias_hh_l{layer}')

        # save weight to load in c
        save_tensor_as_bin(f"rnn_data/weight_ih_l{layer}", w_ih)
        save_tensor_as_bin(f"rnn_data/weight_hh_l{layer}", w_hh)
        save_tensor_as_bin(f"rnn_data/bias_ih_l{layer}", b_ih)
        save_tensor_as_bin(f"rnn_data/bias_hh_l{layer}", b_hh)

        # Compute hidden state
        save_tensor_as_bin("rnn_data/temp1.bin", layer_input)
        save_tensor_as_bin("rnn_data/temp2.bin", w_ih)
        ih = torch.mm(layer_input, w_ih.t())
        save_tensor_as_bin("rnn_data/ih.bin", ih)
        a = torch.mm(layer_input, w_ih.t()) + b_ih + \
            torch.mm(h_t[layer], w_hh.t()) + b_hh
        h_t[layer] = torch.tanh(a)

        # Update input for next layer
        layer_input = h_t[layer]

        # Store the output (which is the hidden state of the last layer)
        outputs[:, t, :] = h_t[-1]

    return outputs, h_t


# Test with PyTorch RNN - let's debug more carefully
set_seed(42)
# Create test input with explicit dimensions
input_size = 10
hidden_size = 20
num_layers = 1
batch_size = 3  # This should match PyTorch's expectation
seq_len = 5

# Create the model
rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

# Create input tensor with correct shape
# (batch_size, seq_len, feature)
input = torch.randn(batch_size, seq_len, input_size)

# Get outputs from PyTorch RNN
torch_output, torch_hn = rnn.forward(input)

# Create directory for weights and data
os.makedirs("rnn_data", exist_ok=True)
# Save input tensor
save_tensor_as_bin("rnn_data/input_data.npy", input)


# Get outputs from custom RNN
custom_output, custom_hn = rnn_custom(input, rnn)
breakpoint()

save_tensor_as_bin("rnn_data/output.bin", custom_output)
save_tensor_as_bin("rnn_data/hn.bin", custom_hn)

# Print shapes for debugging
print("Input shape:", input.shape)
print("PyTorch hidden state shape:", torch_hn.shape)
print("Custom hidden state shape:", custom_hn.shape)

# Verify outputs match
print("PyTorch output shape:", torch_output.shape)
print("Custom output shape:", custom_output.shape)
print("Outputs match:", torch.allclose(
    torch_output, custom_output, rtol=1e-4, atol=1e-4))
print("Hidden states match:", torch.allclose(
    torch_hn, custom_hn, rtol=1e-4, atol=1e-4))
