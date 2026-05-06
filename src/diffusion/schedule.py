"""
Noise schedule for forward process. Contains:

1. linear_schedule
2. cosine_schedule

"""

import math
import torch

def linear_schedule(T,beta_0 = 1e-4,beta_1 = 0.02):
    beta = torch.linspace(beta_0, beta_1, T)
    alpha = 1 - beta 
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta,alpha,alpha_bar

def cosine_schedule(T, beta_max=0.05):
    s = 8e-3
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar_raw = f / f[0]
    alpha = alpha_bar_raw[1:] / alpha_bar_raw[:-1]
    beta = (1 - alpha).clamp(max=beta_max)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta.float(), alpha.float(), alpha_bar.float()