"""
реализация ODE_RNN
"""

import torch.nn as nn
import torch
import torchdiffeq
# from code_folder.Modules import *

class ODE_func(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODE_func, self).__init__()
        self.input_dim = input_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(inplace = True),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, t, x):
        return self.layer(x)


class RNN(nn.Module):
    # gets input of shape (1, batch_size, input_dim)
    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers = num_layers)
    def forward(self, x, h):
        return self.rnn(x, h)


class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralODE, self).__init__()
        self.input_dim = input_dim
        self.func = ODE_func(input_dim, hidden_dim)
    
    def forward(self, x, t, return_whole_sequence = False):
        t = t.squeeze()
        batch_size = x.shape[0]
        device = next(self.parameters()).device
        
# пока действуем в предположении, что времена одинаковы
        out = torchdiffeq.odeint_adjoint(self.func, x, torch.tensor([0, t[0]]).to(device))
        if return_whole_sequence:
            return out
        return out[-1]


class ODE_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 1, return_hidden = False):
        ## input_dim == output_dim
        super(ODE_RNN, self).__init__()
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.rnn = RNN(input_dim, hidden_dim, num_layers)
        self.ode = NeuralODE(hidden_dim * num_layers, hidden_dim * num_layers)
        self.hid2inp = nn.Linear(hidden_dim * num_layers, input_dim)
    def forward(self, x, t):
        #x.shape = (batch_size, input_size, time_steps)
        #t.shape = (batch_size, 1, time_steps)
        #h.shape = (num_layers, batch_size, hidden_dim)
        assert len(x.shape) == 3 
        batch_size = x.shape[0] 
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).float()
        h = h.to(next(self.parameters()).device)
        h_s = []
        for i in range(x.shape[2]):
            x_i = x[..., i ].unsqueeze(0)
            x_i, h = self.rnn(x_i, h)        
            h = h.permute(1, 0 , 2).reshape(batch_size, self.hidden_dim * self.num_layers)
            h = self.ode(h, t[...,i])
            h = h.reshape(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2)
            if self.return_hidden:
                h_s.append(h.detach())
        out = self.hid2inp(h).squeeze()
        return out, h_s , 0


















        