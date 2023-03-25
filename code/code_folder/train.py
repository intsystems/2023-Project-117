from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

# для проверки потока градиентов по модулям
def plot_grad_flow(named_parameters, axs):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            tmp = p.grad.detach().cpu()
            ave_grads.append(tmp.abs().mean())
    axs.plot(ave_grads, alpha=0.3, color="b")
    axs.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    axs.set_xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    axs.set_xlim(xmin=0, xmax=len(ave_grads))
    axs.set_xlabel("Layers")
    axs.set_ylabel("average gradient")
    axs.set_title("Gradient flow")
    axs.grid(True)

def train(net, dataloader,test_dataloader, savepath = None, save_every = 100, optim = None):
##train with iterations
    fig, axs = plt.subplots(1 ,2 , figsize = (10 , 5))
    p = display(fig, display_id = True) 
    axs[0].set_title("L2")
    axs[1].set_title("mean_losses")
    L2 = []
    mean_losses = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr = 0.01, weight_decay = 0.999)

    l2_loss_f = torch.nn.MSELoss()
    # тренируем не по эпохам, а по итерациям
    for i, tmp in tqdm(enumerate(dataloader)):
        data, value, time = tmp
        data = data.float().to(device)
        time = time.float().to(device)
        value = value.float().to(device)

        x_s, h_s, _ = net(data, time)
        x_value = x_s[-1]
        l2_loss = l2_loss_f(x_value, value)
        loss = l2_loss
        optim.zero_grad()
        loss.backward()
        optim.step()

        L2.append(l2_loss.detach().item())
        if i % 5 == 0:
            axs[0].plot(L2)
            p.update(fig)
        if savepath is not None and i % save_every == 0 and i > 0:
            torch.save(net.state_dict(), f"{savepath}_{i}.mod")
# testing every 20 goes
        if i % 20 == 0 and i > 0:

            dat_ , lab_, time_ = test_dataloader                
            dat_ = dat_.float().to(device)
            time_ = time_.float().to(device)
            lab_ = lab_.float().to(device)
            x_s, h_s, _ = net(dat_, time_)
            x_value = x_s[-1]
            # print(x_value, lab_)
            loss = l2_loss_f(x_value, lab_).detach().item()
            mean_loss = loss / dat_.shape[0]
            mean_losses.append(mean_loss)
            axs[1].plot(mean_losses)
            p.update(fig)



def train_with_epoch(net, dataloader,test_dataloader, epochs = 1, optim = None, savepath = None, save_every = 100, save_best = False ):
    fig, axs = plt.subplots(2 ,3 , figsize = (18 , 9))
    p = display(fig, display_id = True) 
    axs[0][0].set_title("L2")
    axs[0][2].set_title("grad_flow")
    axs[0][1].set_title("mean_losses")

    axs[1][0].set_title("L2 epoch loss")
    axs[1][1].set_title("L2 test loss")
    
    axs[1][0].set_xlabel("epochs")
    axs[1][0].set_ylabel("summ_loss")

    axs[1][1].set_xlabel("epochs")
    axs[1][1].set_ylabel("summ_loss")

    L2 = []
    L2_test = []
    L2_epoch_loss = []
    L2_epoch_loss_test = []
    best_test_loss = torch.inf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    if optim is None:
        optim = torch.optim.Adamax(net.parameters(), lr = 1, weight_decay = 0.999)
    
    l2_loss_f = torch.nn.MSELoss()
    
    def train(_dataloader_):
        epoch_loss = 0.
        for i, tmp in tqdm(enumerate(_dataloader_), disable = True):
            data, value, time = tmp
            data = data.float().to(device)
            time = time.float().to(device)
            value = value.float().to(device)
            
            x_s, h_s, internal_loss = net(data, time)
            x_value = x_s[-1]
            l2_loss = l2_loss_f(x_value, value) + internal_loss
            loss = l2_loss
            optim.zero_grad()
            loss.backward()

            plot_grad_flow(net.named_parameters(), axs[1][2])

            optim.step()
            epoch_loss += l2_loss.detach().item()
            L2.append(l2_loss.detach().item())
            if i % 5 == 0:
                axs[0][0].plot(L2)
                p.update(fig)
        axs[0][0].plot(L2)
        L2_epoch_loss.append(epoch_loss)
        axs[1][0].plot(L2_epoch_loss)
        p.update(fig)
        return L2_epoch_loss

    def test(_dataloader_):
        epoch_loss = 0.
        with torch.no_grad():
            for i, tmp in tqdm(enumerate(_dataloader_), disable = True):
                data, value, time = tmp
                data = data.float().to(device)
                time = time.float().to(device)
                value = value.float().to(device)
                
                x_s, h_s, internal_loss = net(data, time)
                x_value = x_s[-1]
                l2_loss = l2_loss_f(x_value, value)  
                epoch_loss += l2_loss.detach().item()              
                L2_test.append(l2_loss.detach().item())
        axs[0][1].plot(L2_test)
        L2_epoch_loss_test.append(epoch_loss)
        axs[1][1].plot(L2_epoch_loss_test)
        p.update(fig)
        return epoch_loss


    for i, tmp in enumerate( zip(dataloader, test_dataloader)):
        train(tmp[0])
        epoch_loss = test(tmp[1])
        if save_best and savepath is not None and epoch_loss <= best_test_loss:
            best_test_loss = epoch_loss
            torch.save(net.state_dict(), f"{savepath}_best_model_{epoch_loss}.mod")  
    return net     

  

           