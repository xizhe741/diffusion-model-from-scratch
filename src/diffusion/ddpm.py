import torch
class Gaussian_diff:
    def __init__(self,alpha_bar):
        self.alpha_bar = alpha_bar
        self.T = len(self.alpha_bar) 
    def compute_loss(self,net,x0):
        B = x0.shape[0]
        t = torch.randint(0,self.T,(B,),device = x0.device)
        xt,noise = self.q_sample(x0,t)
        noise_predict = net(xt,t)
        loss = ((noise - noise_predict)**2).mean()
        return loss
    def q_sample(self,x0,t,noise =None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t.cpu()].to(x0.device).view(-1,1,1,1)
        xt = torch.sqrt(alpha_bar_t)*x0\
        + torch.sqrt(1-alpha_bar_t)*noise
        return xt,noise