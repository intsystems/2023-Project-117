"""
The given code defines several neural network classes.
Here is a brief description of each class and its purpose:

1. ODE_func: A class that defines an ordinary differential equation (ODE) 
function as a neural network. The ODE function takes two inputs: time t and a state vector x,
and returns the evolutuion of the state vector with respect to time.

2. ODE_RNNencoder: A class that defines an encoder network that takes a time series data
with shape (batch_size, input_size, time_steps) and returns the mean and the log standard 
deviation of the latent representation calculated from the final hidden state
of a GRU layer applied to the data. 
The encoder also returns the hidden states at each time step if the `return_hidden` parameter is set to True.

3. ODEdecoder: A class that defines a decoder network that takes a 
latent representation and returns the reconstructed time series data.

4. ODERNN: A class that defines the main model, which combines the encoder
 and decoder networks to create a variational autoencoder (VAE) that can encode and decode time series data.

"""

import torch.nn as nn
import torch
import torchdiffeq
from Nodules import * # RNN и NeuralODE берутся отсюда
class RNN(nn.Module):
    # gets input of shape (1, batch_size, input_dim)
    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers = num_layers)
    def forward(self, x, h):
        return self.rnn(x, h)

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

class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NeuralODE, self).__init__()
        self.input_dim = input_dim
        self.func = ODE_func(input_dim, hidden_dim)
    def forward(self, x, t, return_whole_sequence = False):
        out = torchdiffeq.odeint_adjoint(self.func, x, t.squeeze())
        if return_whole_sequence:
            return out
        return out[-1]

class ODE_RNNencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers = 1, return_hidden = False):
        ## input_dim == output_dim
        super(ODE_RNNencoder, self).__init__()
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rnn = RNN(input_dim, hidden_dim, num_layers)
        self.ode = NeuralODE(hidden_dim * num_layers, hidden_dim * num_layers)
        self.hid2lat = nn.Linear(hidden_dim * num_layers, 2 *latent_dim)
    def forward(self, x, t):
        #x.shape = (batch_size, input_size, time_steps)
        #t.shape = (batch_size, 1, time_steps)
        #h.shape = (num_layers, batch_size, hidden_dim)
        assert len(x.shape) == 3 
        batch_size = x.shape[0] 
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).float()
        h = h.to(next(self.parameters()).device)
        x = x.flip((2,))
        t = t.flip((2,))
        h_s = []
        for i in range(x.shape[2] - 1):
            x_i = x[..., i ].unsqueeze(0)
            _, h = self.rnn(x_i, h)
            h = h.permute(1, 0 , 2).reshape(batch_size, self.hidden_dim * self.num_layers)
            h = self.ode(h, t[...,i: i + 2])
            h = h.reshape(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2)
            if self.return_hidden:
                h_s.append(h.detach())
        h = h.permute(1, 0 , 2).reshape(batch_size, self.hidden_dim * self.num_layers)
        h = self.hid2lat(h) ## shape: (batch_size, 2 * latent_dim)
        mu = h[:, :self.latent_dim]
        log_sigma = h[:, self.latent_dim:]
        return mu, log_sigma, h_s 

class ODEdecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.ode = NeuralODE(latent_dim, 2 * latent_dim)
        self.lat2in = nn.Linear(latent_dim, input_dim)
    def forward(self, z, t):
        z = self.ode(z, t, return_whole_sequence = True)
        z = self.lat2in(z)
        return z

class ODERNN(nn.Module):
    """
    Основная  модель
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers = 1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.latend_dim = latent_dim
        self.encoder = ODE_RNNencoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = ODEdecoder(input_dim, hidden_dim, latent_dim)
    def forward(self, x, t, MAP = False):
        mu, log_sigma, _ = self.encoder(x, t)
        if MAP:
            z = mu
        else:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_sigma)
        x_s = self.decoder(z, t)
        x_s = x_s.permute(1,2,0)
        return x_s, mu, log_sigma
    def decode_with_hidden(self, x, t):
        self.eval()
        self.encoder.return_hidden = True
        mu, log_sigma, h_s = self.encoder(x, t)
        self.encoder.return_hidden = False        
        return mu, log_sigma, h_s
    def decode(self, x, t):
        self.eval()
        mu, log_sigma, _ = self.encoder(x, t)
        return mu, log_sigma


class KL_loss(nn.Module):
    def __init__(self, unnealing = 0.99):
        self.unnealing = unnealing
        self.unneal_coef = 1.
        super().__init__()
    def forward(self, mu, log_sigma):
        self.unneal_coef *= self.unnealing 
        return self.unneal_coef * (-0.5) * torch.sum(1 + log_sigma - mu**2 - torch.exp(log_sigma), -1)