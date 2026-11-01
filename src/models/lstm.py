import torch
import torch.nn as nn


class LSTMprev(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.input_x = nn.Linear(input_size, hidden_size)
        self.input_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.forget_x = nn.Linear(input_size, hidden_size)
        self.forget_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.output_x = nn.Linear(input_size, hidden_size)
        self.output_h = nn.Linear(hidden_size, hidden_size, bias=False)

        self.candidate_x = nn.Linear(input_size, hidden_size)
        self.candidate_h = nn.Linear(hidden_size, hidden_size, bias=False)

        nn.init.ones_(self.forget_x.bias)

    def forward(self, x, state):
        h_prev, c_prev = state

        i_t = torch.sigmoid(self.input_x(x) + self.input_h(h_prev))
        f_t = torch.sigmoid(self.forget_x(x) + self.forget_h(h_prev))
        o_t = torch.sigmoid(self.output_x(x) + self.output_h(h_prev))
        c_hat = torch.tanh(self.candidate_x(x) + self.candidate_h(h_prev))

        c_t = f_t * c_prev + i_t * c_hat
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.cell = LSTMprev(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)

        for t in range(seq_length):
            h, c = self.cell(x[:, t, :], (h, c))

        return self.output_layer(h)