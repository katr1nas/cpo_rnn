import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(
            hidden_size, hidden_size, bias=False
        )

    def forward(self, x, h_prev):
        h = torch.tanh(
            self.input_layer(x) + self.hidden_layer(h_prev)
        )
        return h
    

class RNN(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)

        for t in range(seq_length):
            h = self.cell(x[:, t, :], h)

        output = self.output_layer(h)

        return output