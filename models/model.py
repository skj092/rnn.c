from torch import mode, nn
import torch
import numpy as np
from rnn import set_seed, save_tensor_as_bin


# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        save_tensor_as_bin('modeltest/rnn_out', rnn_out)
        save_tensor_as_bin('modeltest/hidden', hidden)
        output = self.h2o(hidden[0])
        save_tensor_as_bin("modeltest/lo", output)
        breakpoint()
        output = self.softmax(output)

        return output


set_seed(42)
n_input = 26
n_hidden = 128
n_output = 10

model = CharRNN(n_input, n_hidden, n_output)
x = torch.randn(9, 1, 26)
save_tensor_as_bin("modeltest/x.bin", x)
out = model(x)
breakpoint()
save_tensor_as_bin("modeltest/out.bin", out)

# save model weight and bias for reproducibility in c
save_tensor_as_bin("modeltest/weight_ih_l0", model.rnn.weight_ih_l0)
save_tensor_as_bin("modeltest/weight_hh_l0", model.rnn.weight_hh_l0)
save_tensor_as_bin("modeltest/bias_ih_l0", model.rnn.bias_ih_l0)
save_tensor_as_bin("modeltest/bias_hh_l0", model.rnn.bias_hh_l0)
save_tensor_as_bin("modeltest/lw", model.h2o.weight.data)
save_tensor_as_bin("modeltest/lb", model.h2o.bias.data)
