import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F
from collections import defaultdict


class RNN_ODE_func(nn.Module):
    def compose_x_with_h(self, x, h, num_layers, device):
        x = torch.cat([x, *[h[i] for i in range(num_layers)]], dim = -1)
        return x.to(device)

    def decompose_x_with_h(self, inp, input_dim, hidden_dim, num_layers, device):
        inp = inp.unsqueeze(0)
        x =  inp[..., :input_dim].to(device)
        h_s = torch.cat( 
                [ inp[..., input_dim + hidden_dim * i 
                        :input_dim + hidden_dim * (i + 1) ] 
                        for i in range(num_layers)], dim = 0 )
        h_s = h_s.to(device)
        return x, h_s

    def __init__(self, input_dim, hidden_dim, num_layers = 1):
        super(RNN_ODE_func, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.hid2inp = nn.Linear(hidden_dim, input_dim)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers = num_layers)
        self.norm = nn.LayerNorm((num_layers, hidden_dim))
        self.out_norm = nn.LayerNorm((input_dim))
        
    def forward(self,t, x):
        """
        t - Время, но мы его не используем
        x.shape = (batch_size, input_dim + hidden_dim * num_layers)
        """
        device = next(self.parameters()).device
    # unsqueeze(0) чтобы добавить количество шагов по x у
        input_ , h_s = self.decompose_x_with_h(x, self.input_dim , self.hidden_dim, self.num_layers, device)
        output_, h_s = self.rnn(input_, h_s)
        output_ = output_.view(output_.shape[1:])
        # h_s = self.layer(h_s)
    # если батч был размера 1 то его размерность была убрана на предыдущем шаге
        output_ = self.hid2inp(output_)
        h_s = self.norm(h_s.permute(1, 0, 2)).permute(1, 0, 2)
        output_ = self.out_norm(output_)
        rez = self.compose_x_with_h(output_, h_s, self.num_layers, device)
        return rez

# хотелосб изначально сделать одинаковые обработуи чтобы подавать как модули 
class LSTM_ODE_func(nn.Module):
    def compose_x_with_h(self, x, h,c, num_layers, device):
        x = torch.cat([x, *[h[i] for i in range(num_layers)],*[c[i] for i in range(num_layers)]], dim = -1)
        return x.to(device)

    def decompose_x_with_h(self, inp, input_dim, hidden_dim, cell_dim, num_layers, device):
        inp = inp.unsqueeze(0)
        x = inp[..., : input_dim].to(device)
        h_s = torch.cat( 
                [ inp[..., input_dim + hidden_dim * i 
                        :input_dim + hidden_dim * (i + 1) ] 
                        for i in range(num_layers)], dim = 0 )
        start_pos = input_dim + hidden_dim * num_layers
        c_s = torch.cat(
            [ inp[..., start_pos + cell_dim* i:
                    start_pos + cell_dim*(i + 1)]
                    for i in range(num_layers)], dim = 0 )
        return x, h_s, c_s
    
    def __init__(self, input_dim, hidden_dim, proj_dim = 0, num_layers = 1):
        super(LSTM_ODE_func, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = proj_dim if proj_dim > 0 else hidden_dim
        self.num_layers = num_layers
        self.out2inp = nn.Linear(self.out_dim, input_dim)
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, proj_size = self.out_dim)
    def forward(self, t, x):
        device = next(self.parameters()).device
        input_, h_s, c_s = self.decompose_x_with_h(x, self.input_dim , self.out_dim, self.hidden_dim, self.num_layers, device)
        output_, (h_s, c_s) = self.rnn(input_ , (h_s, c_s))
        output_ = output_.view(output_.shape[1:])
        output_ = self.out2inp(output_)
        return self.compose_x_with_h(output_, h_s, c_s, self.num_layers, device)




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
 
 
####################################################################################
    #         sorter_struct = zip(list(range(batch_size)), x, t)
    #         sorter_struct = sorted(sorter_struct, key = lambda key_ : key_[2])
    #         x_ = torch.vstack([sorter_struct[i][1] for i in range(batch_size)])
    #         x_ = x_.to(device)
    #     # нужно выкинуть одинаковые промежутки интегрирования
    #         t_ = [sorter_struct[0][2].item()]
    #         i = 0
    #         for j in range(1, batch_size):
    #             if abs(sorter_struct[i][2].item() - sorter_struct[j][2].item()) > self.tolerance:
    #                 t_.append(sorter_struct[j][2].item())
    #                 i = j         
    #         t_ = torch.tensor([0., *t_]).to(device)
    #         # out.shape = (len(t_), batch_size, hidden_dim)
    #         # в нулевой координате содержится X_ не интегрированный. Он нам не нужен
            
    #         out = torchdiffeq.odeint_adjoint(self.func, x_, t_, method = 'euler')
    #         # print(out.shape, t_.shape)

    #         out = out[1:]
    #         out_new = []
    # # отбираем нужные промежутки интегрирования
    #         j = 0
    #         for i, tt in enumerate(t_[1:]):
    #             while abs(sorter_struct[j][2] - tt) <= self.tolerance:
    #                 out_new.append(out[i][j])
    #                 j += 1
    #                 if j == batch_size:
    #                     break
    #         out = torch.vstack(out_new)
    #         # теперь нужно восстановить порядок элементов батча. Это делатся по индексам в sorter_struct 
    #         sorter_struct = zip([sorter_struct[i][0] for i in range(batch_size)], out)
    #         sorter_struct = sorted(sorter_struct, key = lambda key_: key_[0])
    #         out = torch.vstack([sorter_struct[i][1] for i in range(batch_size)])
    #         out = out.to(device)
#####################################################################################
#  то же но быстрее
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
            out = torchdiffeq.odeint_adjoint(self.func, x, t_, method = 'euler', rtol = 1e-8)
            indices = []
            for time_index, elem in enumerate(t_dict.items()):
                ind = elem[1]        
                for elem_index in ind:
                    indices.append([time_index, elem_index])
            indices = torch.tensor(sorted(indices, key = lambda key_: key_[1])).T
            out = out[[indices[0],indices[1]]]
#################################################################################       
        return out


class ODE_on_RNN(nn.Module):
    def __init__(self,  input_dim, hidden_dim, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_on_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ode_func = RNN_ODE_func(input_dim, hidden_dim, num_layers)
        self.ode = NeuralODE(self.ode_func, same_intervals, tolerance)
        self.loss_function = nn.MSELoss()
        self.internal_loss_dim = internal_loss_dim
        
    # для инициалии h_0
        self.inp2hid = nn.Linear(input_dim, hidden_dim * num_layers)
    def forward(self, x, t, return_hidden = False):
        # x.shape = (batch_size, input_shape, time_steps)
        #t.shape = (batch_size, 1, time_steps)
        # h.shape = (num_layers, batch_size, hidden_dim)
        internal_loss = 0
        scale_factor = 0.1
        device = next(self.parameters()).device
        batch_size = x.shape[0]


        h = self.inp2hid(x[..., 0])
        h = torch.cat([h[..., self.hidden_dim * i : self.hidden_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        h_s = []
        for i in range(x.shape[2]):
            if return_hidden:
                h_s.append(h.detach())
            x_i = x[..., i]
            inp = self.ode_func.compose_x_with_h(x_i, h, self.num_layers, device)
            out = self.ode(inp, t[..., i])
            out, h = self.ode_func.decompose_x_with_h(out,  self.input_dim , self.hidden_dim, self.num_layers, device)
            if self.internal_loss_dim is not None and i != x.shape[2] - 1:
                internal_loss += self.loss_function(out[..., : self.internal_loss_dim], x[..., i + 1][..., :self.internal_loss_dim]) * scale_factor
                scale_factor = scale_factor * 1.5
        return out, h_s, internal_loss 
            


class ODE_on_LSTM(nn.Module):
    def __init__(self,  input_dim, hidden_dim, proj_dim = 0, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_on_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ode_func = LSTM_ODE_func(input_dim, hidden_dim, proj_dim, num_layers)
        self.ode = NeuralODE(self.ode_func, same_intervals, tolerance)
        self.loss_function = nn.MSELoss()
        self.internal_loss_dim = internal_loss_dim

        self.out_dim = self.ode_func.out_dim
    # для инициалии h_0
        self.inp2hid = nn.Linear(input_dim, hidden_dim * num_layers)
        self.inp2out = nn.Linear(input_dim, self.out_dim * num_layers)
    def forward(self, x, t, return_hidden = False):
        # x.shape = (batch_size, input_shape, time_steps)
        #t.shape = (batch_size, 1, time_steps)
        # h.shape = (num_layers, batch_size, hidden_dim)
        internal_loss = 0

        device = next(self.parameters()).device
        batch_size = x.shape[0]
        c = self.inp2hid(x[..., 0])
        c = torch.cat([c[..., self.hidden_dim * i : self.hidden_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        h = self.inp2out(x[..., 0])
        h = torch.cat([h[..., self.out_dim * i : self.out_dim * (i + 1)].unsqueeze(0) 
            for i in range(self.num_layers)], dim = 0)
        
        for i in range(x.shape[2]):
            x_i = x[..., i]
            inp = self.ode_func.compose_x_with_h(x_i, h, c,  self.num_layers, device)
            out = self.ode(inp, t[..., i])
            out, h, c = self.ode_func.decompose_x_with_h(out, self.input_dim , self.out_dim, self.hidden_dim, self.num_layers, device)
            if self.internal_loss_dim is not None and i != x.shape[2] - 1:
                # print(out[..., : self.internal_loss_dim].shape,   x[..., i + 1][..., :self.internal_loss_dim] .shape)
                internal_loss += self.loss_function(out[..., : self.internal_loss_dim], x[..., i + 1][..., :self.internal_loss_dim])
        return out, 0, internal_loss


class ODE_on_RNN_featured(nn.Module):
    def __init__(self, input_dim, return_dim, hidden_dim, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_on_RNN_featured, self).__init__()
        assert return_dim <= input_dim
        self.net = ODE_on_RNN(input_dim, hidden_dim, num_layers, same_intervals, tolerance, internal_loss_dim)
        self.return_dim = return_dim
    def forward(self, x, t, return_hidden = False):
        out, h_s, loss = self.net(x, t , return_hidden)
        return out[..., :self.return_dim], h_s, loss 

class ODE_on_LSTM_featured(nn.Module):
    def __init__(self, input_dim, return_dim, hidden_dim, proj_dim = 0, num_layers = 1, same_intervals = True, tolerance = 1e-5, internal_loss_dim = None):
        super(ODE_on_LSTM_featured, self).__init__()
        assert return_dim <= input_dim
        self.net = ODE_on_LSTM(input_dim, hidden_dim, proj_dim, num_layers, same_intervals, tolerance, internal_loss_dim)
        self.return_dim = return_dim
    def forward(self, x, t, return_hidden = False):
        out, h_s, loss = self.net(x, t , return_hidden)
        return out[..., :self.return_dim], h_s, loss 






