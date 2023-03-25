"""
модули общие для остальных модейлей

The given code defines several neural network classes.
Here is a brief description of each class and its purpose:

1. RNN: A recurrent neural network class that takes an input of shape (1, batch_size, input_dim),
 and returns the output of a GRU layer applied to the input.

2. NeuralODE: A class that defines a neural ODE, which applies the ODE function 
defined in ODE_func to a given input x and returns the output of the ODE integration.

"""
import torch.nn as nn
import torch
import torchdiffeq
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
        #t.shape = (batch_size)
        # в качестве t подаются времена на которые нужно эволюционировать каждый элемент батча
        # поэтому добавляем к временам 0, сортируем по delta_t и применяем то что было у ментора
        # out.shape = (batch_size, batch_size, hidden_dim)
        
        # sorter_struct = zip(list(range(batch_size)), x, t)
        # sorter_struct = sorted(sorter_struct, key = lambda key_ : key_[2])
        # x_ = torch.vstack([sorter_struct[i][1] for i in range(batch_size)])
        # x_ = x_.to(device)
        # t_ = torch.tensor([0., *[sorter_struct[i][2] for i in range(batch_size)]])
        # t_ = t_.to(device)
        # print(t)
        # print(t_)
        
        # out.shape = (batch_size + 1, batch_size, hidden_dim)
        #в нулевой координате содержится X_ не интегрированный. Он нам не нужен
        
        # out = torchdiffeq.odeint_adjoint(self.func, x_, t_)
        # out = out[1:]
        # out = torch.diagonal(out).T
        
        # теперь нужно восстановить порядок элементов батча. Это делатся по индексам в sorter_struct
        
        # sorter_struct = zip([sorter_struct[i][0] for i in range(batch_size)], out)
        # sorter_struct = sorted(sorter_struct, key = lambda key_: key_[0])
        # out = torch.vstack([sorter_struct[i][1] for i in range(batch_size)])
        # out = out.to(device)
        
        if return_whole_sequence:
            return out
        return out[-1]






