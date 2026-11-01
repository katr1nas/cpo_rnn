import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class GRUprev(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.update_x = nn.Linear(input_size, hidden_size)
        self.update_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.reset_x = nn.Linear(input_size, hidden_size)
        self.reset_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.candidate_x = nn.Linear(input_size, hidden_size)
        self.candidate_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        z_t = torch.sigmoid(self.update_x(x) + self.update_h(h_prev))
        r_t = torch.sigmoid(self.reset_x(x) + self.reset_h(h_prev))
        h_hat = torch.tanh(self.candidate_x(x) + self.candidate_h(r_t * h_prev))

        return (1 - z_t) * h_prev + z_t * h_hat
    

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.cell = GRUprev(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)

        for t in range(seq_length):
            h = self.cell(x[:, t, :], h)

        output = self.output_layer(h)

        return output
        
