import torch

def q_sample(x0,t,alpha_bar,noise =None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bar[t].view(-1,1,1,1)
    xt = torch.sqrt(alpha_bar_t)*x0\
    + torch.sqrt(1-alpha_bar_t)*noise
    return xt,noise