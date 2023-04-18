"""
модули общие для остальных модейлей

The given code defines several neural network classes.
Here is a brief description of each class and its purpose:

1. RNN: A recurrent neural network class that takes an input of shape (1, batch_size, input_dim),
 and returns the output of a GRU layer applied to the input.

2. NeuralODE: A class that defines a neural ODE, which applies the ODE function 
defined in ODE_func to a given input x and returns the output of the ODE integration.

"""
import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F
from collections import defaultdict
import torchcde as cde

class Hooks:
    def add_hooks(self):
        for p in self.ode_func.parameters():
            p.register_hook(lambda grad: grad / torch.norm(grad))
            
class NeuralODE(nn.Module):
    def __init__(self, ode_func, same_intervals = True, tolerance = 1e-5):
        super(NeuralODE, self).__init__()
        self.func = ode_func
        self.same_intervals = same_intervals
        self.tolerance = tolerance
    
    def forward(self, x, t):
        # x.shape = (batch_size, ...)
        t = t.view(t.shape[0])
        batch_size = x.shape[0]
        device = next(self.parameters()).device
        
        if self.same_intervals:
        # действия в предположении, что времена одинаковы
            t1 = t[0] if t.shape[0] > 1 else t.item()
            out = torchdiffeq.odeint_adjoint(self.func, x, torch.tensor([0, t1]).to(device), method = 'euler')
            out = out[-1]
            return out
        else:
        #действия в случае различных времен интегрирования
        # сортируем по времени интегрирования запоминая порядок
        # отбираем нужные и востанавливаем порядок             
            sorter_struct = zip(list(range(batch_size)), t)
            sorter_struct = sorted(sorter_struct, key = lambda key_ : key_[1])
        # нужно выкинуть одинаковые промежутки интегрирования
            t_ = [torch.zeros(1)]
            t_dict = defaultdict(list)
            t_dict[0] = []
            for j in range(0, batch_size):
                if abs(t_[-1] - sorter_struct[j][1]) > self.tolerance:
                    t_.append(sorter_struct[j][1])
                t_dict[t_[-1].item()].append(sorter_struct[j][0])    

            t_ = torch.hstack(t_).to(device)
            # out.shape = (len(t_), batch_size, hidden_dim)
            # в нулевой координате содержится X_ не интегрированный. Он нам не нужен            
            out = torchdiffeq.odeint_adjoint(self.func, x, t_, method = 'midpoint')
            indices = []
            for time_index, elem in enumerate(t_dict.items()):
                ind = elem[1]        
                for elem_index in ind:
                    indices.append([time_index, elem_index])
            indices = torch.tensor(sorted(indices, key = lambda key_: key_[1])).T
            out = out[[indices[0],indices[1]]]
#################################################################################       
        return out

class ode_func_interface(nn.Module):
    def __init__(self, ): 
        super(ode_func_interface, self).__init__()
        self.device = None  
    def compose_x_with_h(self, x, h): raise NotImplementedError()
    def decompose_x_with_h(self, inp): raise NotImplementedError()
    def inp2hid(self, t, x): raise NotImplementedError()
    def forward(self, t , x): raise NotImplementedError()

class GRU_ODE_func(ode_func_interface):
    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super(GRU_ODE_func, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hid2inp = nn.Linear(hidden_dim, input_dim)
        self._inp2hid_ = nn.Linear(input_dim, hidden_dim * num_layers)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers = num_layers)
        self.norm = nn.LayerNorm((num_layers, hidden_dim))
        self.device =  None
    def compose_x_with_h(self, x, h):
        x = torch.cat([x, *[h[i] for i in range(self.num_layers)]], dim = -1)
        return x.to(self.device)
    def decompose_x_with_h(self, inp):
        inp = inp.unsqueeze(0)
        x = inp[..., : self.input_dim].to(self.device)
        h_s = torch.cat(
            [inp[..., self.input_dim + self.hidden_dim * i:
                    self.input_dim + self.hidden_dim * ( i + 1)]
                        for i in range(self.num_layers) ] , dim = 0 )
        h_s = h_s.to(self.device)
        return x, [h_s]
    def inp2hid(self, t, x):
        #print(x.shape, self.input_dim, self.hidden_dim)
        h = self._inp2hid_(x[..., 0])
        h = torch.cat([h[..., self.hidden_dim * i : self.hidden_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        return [h]
    def forward(self, t, x):
        input_, h_s = self.decompose_x_with_h(x)
        output_, h_s = self.rnn(input_, *h_s)
        output_ = output_.view(output_.shape[1:])
        output_ = self.hid2inp(output_)
        h_s = self.norm(h_s.permute(1, 0, 2)).permute(1, 0, 2)
        return self.compose_x_with_h(output_, h_s)

class GRU_ODE_INTERPOLATE_func(GRU_ODE_func):
    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super(GRU_ODE_INTERPOLATE_func, self).__init__(input_dim, hidden_dim, num_layers)
        self.X = None
        self.t = None
        self.t_index = None
        self.border = None

    def inp2hid(self, t, x):
        x_ = torch.permute(x, (0, 2, 1))
        t_ = t[0].squeeze()
        t_ = torch.cat([torch.zeros(1), t_], dim = 0)[:-1]
        t_ = torch.cumsum(t_, dim = 0)
        coeffs = cde.hermite_cubic_coefficients_with_backward_differences(x_, t_)
        self.X = cde.CubicSpline(coeffs, t_)
        self.t = t_
        self.t_index = -1
        self.border = float("inf") 
        h = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim).float().to(device)  
        # h = self._inp2hid_(x[..., 0])
        # h = torch.cat([h[..., self.hidden_dim * i : self.hidden_dim * (i + 1)].unsqueeze(0) 
        #     for i in range(self.num_layers)], dim = 0)
        return [h]
    
    def get_x(self, t):
        if t < self.border: self.t_index += 1
        tmp = self.X.evaluate(self.t[self.t_index] + t)
        self.border = t.item()
        return tmp.unsqueeze(0).to(self.device)
        
    def forward(self, t, x):
        _, h_s = self.decompose_x_with_h(x)
        input_ = self.get_x(t)
        output_, h_s = self.rnn(input_, *h_s)
        output_ = output_.view(output_.shape[1:])
        output_ = self.hid2inp(output_)
        h_s = self.norm(h_s.permute(1, 0, 2)).permute(1, 0, 2)
        return self.compose_x_with_h(output_, h_s)


class LSTM_ODE_func(ode_func_interface):
    def __init__(self, input_dim, hidden_dim, proj_dim = 0, num_layers = 1):
        super(LSTM_ODE_func, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = proj_dim if proj_dim > 0 else hidden_dim - 1
        self.num_layers = num_layers
        self.out2inp = nn.Linear(self.out_dim, input_dim)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, proj_size = self.out_dim)
        self.device = torch.device("cpu")
        self._inp2hid_ = nn.Linear(input_dim, hidden_dim * num_layers)
        self.inp2out = nn.Linear(input_dim, self.out_dim * num_layers)
    
    def compose_x_with_h(self, x, h,c):
        x = torch.cat([x, *[h[i] for i in range(self.num_layers)],*[c[i] for i in range(self.num_layers)]], dim = -1)
        return x.to(self.device)

    def decompose_x_with_h(self, inp):
        inp = inp.unsqueeze(0)
        x = inp[..., : self.input_dim].to(self.device)
        h_s = torch.cat( 
                [ inp[..., self.input_dim + self.out_dim * i 
                        :self.input_dim + self.out_dim * (i + 1) ] 
                        for i in range(self.num_layers)], dim = 0 )
        start_pos = self.input_dim + self.out_dim * self.num_layers
        c_s = torch.cat(
            [ inp[..., start_pos + self.hidden_dim* i:
                    start_pos + self.hidden_dim*(i + 1)]
                    for i in range(self.num_layers)], dim = 0 )
        return x, [h_s.to(self.device), c_s.to(self.device)]
    
    def inp2hid(self, t, x):
        c = self._inp2hid_(x[..., 0])
        c = torch.cat([c[..., self.hidden_dim * i : self.hidden_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        h = self.inp2out(x[..., 0])
        h = torch.cat([h[..., self.out_dim * i : self.out_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        return [h.to(self.device), c.to(self.device)]

    def forward(self, t, x):
        device = next(self.parameters()).device
        input_, [h_s, c_s] = self.decompose_x_with_h(x)
        output_, (h_s, c_s) = self.rnn(input_ , (h_s, c_s))
        output_ = output_.view(output_.shape[1:])
        output_ = self.out2inp(output_)
        return self.compose_x_with_h(output_, h_s, c_s)




class ODE_on_RNN(nn.Module, Hooks):
    def __init__(self, ode_func, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_on_RNN, self).__init__()
        self.ode_func = ode_func
        self.ode = NeuralODE(self.ode_func, same_intervals, tolerance)
        self.loss_function = nn.MSELoss()
        self.internal_loss_dim = internal_loss_dim
    

    def forward(self, x, t, return_hidden = False):
        internal_loss = 0
        device = next(self.parameters()).device
        self.ode_func.device = device
        h = self.ode_func.inp2hid(t ,x)
        h_s = []
        for i in range(x.shape[2]):
            if return_hidden:
                h_s.append(h)
            x_i = x[..., i]
            inp = self.ode_func.compose_x_with_h(x_i, *h)
            out = self.ode(inp, t[..., i])
            out, h = self.ode_func.decompose_x_with_h(out)
            if self.internal_loss_dim is not None and i != x.shape[2] -1 :
                internal_loss += self.loss_function(out[..., : self.internal_loss_dim], 
                                                    x[..., i + 1][..., :self.internal_loss_dim])
        return out, h_s, internal_loss


class ODE_RNN(nn.Module, Hooks):
    class ODE_func(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(ODE_RNN.ODE_func, self).__init__()
            self.ode_func = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(inplace = True),
                nn.Linear(hidden_dim, input_dim)
            )
        def forward(self, t, x):
            return self.ode_func(x)

    def __init__(self, input_dim, hidden_dim , num_layers = 1, next_hidden_dim = None, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_RNN, self).__init__()
        if next_hidden_dim is None:
            next_hidden_dim = num_layers * hidden_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers = num_layers)
        self.ode_func =  ODE_RNN.ODE_func(hidden_dim * num_layers, next_hidden_dim)
        self.ode = NeuralODE(self.ode_func,  same_intervals, tolerance)
        self.hid2inp = nn.Linear(hidden_dim * num_layers, input_dim)
        self.internal_loss_dim = internal_loss_dim
        self.loss_function = nn.MSELoss()

    def forward(self, x, t, return_hidden = False):
        assert len(x.shape) == 3
        device = next(self.parameters()).device
        internal_loss = 0.
        batch_size = x.shape[0]
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).float().to(device)
        h_s = []
        for i in range(x.shape[2]):
            x_i = x[..., i].unsqueeze(0)
            _, h = self.rnn(x_i, h)
            h = h.permute(1, 0 , 2).reshape(batch_size, self.hidden_dim * self.num_layers)
            h = self.ode(h, t[...,i])
            h = h.reshape(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2)
            if return_hidden:
                h_s.append([h])
            if self.internal_loss_dim is not None and i != x.shape[2] -1:
                out = self.hid2inp(h)
                internal_loss += self.loss_function(out, x[..., i + 1][..., :self.internal_loss_dim])
        out = self.hid2inp(h)
        return out, h_s, internal_loss

        out = self.hid2inp(h).squeeze()
        return out, [h_s] , internal_loss


def ode_on_gru(input_dim, hidden_dim, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
    ode_func = GRU_ODE_func(input_dim, hidden_dim, num_layers = 1)
    net = ODE_on_RNN(ode_func,  same_intervals, tolerance, internal_loss_dim)
    return net

def ode_on_gru_interpolate(input_dim, hidden_dim, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
    ode_func = GRU_ODE_INTERPOLATE_func(input_dim, hidden_dim, num_layers = 1)
    net = ODE_on_RNN(ode_func,  same_intervals, tolerance, internal_loss_dim)
    return net

def ode_on_lstm( input_dim, hidden_dim, proj_dim = 0, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
    ode_func = LSTM_ODE_func( input_dim, hidden_dim, proj_dim, num_layers)
    net = ODE_on_RNN(ode_func,  same_intervals, tolerance, internal_loss_dim)
    return net


from copy import deepcopy
class ODE_on_RNN_diff(nn.Module, Hooks):
    def __init__(self, ode_func, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_on_RNN_diff, self).__init__()
        self.ode_func = ode_func
        self.ode = NeuralODE(self.ode_func, same_intervals, tolerance)
        self.read_ode = nn.Sequential(
            nn.Linear(ode_func.input_dim  + ode_func.hidden_dim * ode_func.num_layers, ode_func.input_dim  + ode_func.hidden_dim * ode_func.num_layers),
            nn.ELU(),
            nn.Linear(ode_func.input_dim  + ode_func.hidden_dim * ode_func.num_layers, ode_func.input_dim  + ode_func.hidden_dim * ode_func.num_layers),
        )#deepcopy(ode_func)
        self.loss_function = nn.MSELoss()
        self.internal_loss_dim = internal_loss_dim
    

    def forward(self, x, t, return_hidden = False):
        internal_loss = 0
        device = next(self.parameters()).device
        self.ode_func.device = device
        h = self.ode_func.inp2hid(t ,x)
        h_s = []
        for i in range(x.shape[2]):
            if return_hidden:
                h_s.append(h)
            x_i = x[..., i]
            inp = self.ode_func.compose_x_with_h(x_i, *h)
            inp = self.read_ode(inp)
            out = self.ode(inp, t[..., i])
            out, h = self.ode_func.decompose_x_with_h(out)
            if self.internal_loss_dim is not None and i != x.shape[2] -1 :
                internal_loss += self.loss_function(out[..., : self.internal_loss_dim], 
                                                    x[..., i + 1][..., :self.internal_loss_dim])
        return out, h_s, internal_loss

def ode_on_gru_diff(input_dim, hidden_dim, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
    ode_func = GRU_ODE_func(input_dim, hidden_dim, num_layers = 1)
    net = ODE_on_RNN_diff(ode_func,  same_intervals, tolerance, internal_loss_dim)
    return net

def ode_on_gru_interpolate_diff(input_dim, hidden_dim, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
    ode_func = GRU_ODE_INTERPOLATE_func(input_dim, hidden_dim, num_layers = 1)
    net = ODE_on_RNN_diff(ode_func,  same_intervals, tolerance, internal_loss_dim)
    return net

def ode_on_lstm_diff( input_dim, hidden_dim, proj_dim = 0, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
    ode_func = LSTM_ODE_func( input_dim, hidden_dim, proj_dim, num_layers)
    net = ODE_on_RNN_diff(ode_func,  same_intervals, tolerance, internal_loss_dim)
    return net

   





