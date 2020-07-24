# https://github.com/seba-1511/lstms.pth/blob/master/lstms/lstm.py
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
import math

class LayerNormLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormLSTMCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(4*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(4*hidden_size, elementwise_affine=False)
        self.ln_cell = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.i2h = torch.nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.hidden_size=hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h, c):
        # h, c = hidden
        h = h
        c = c
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size] 
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        # cell computations
        c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

        # Layer norm
        c_t = self.ln_cell(c_t)

        h_t = torch.mul(o_t, c_t.tanh())

        # Reshape for compatibility

        h_t = h_t.view( h_t.size(0), -1)
        c_t = c_t.view( c_t.size(0), -1)
        return h_t, c_t