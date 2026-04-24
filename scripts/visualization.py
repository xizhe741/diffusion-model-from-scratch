import torch
import torchvision
import torchvision.transforms as T

import matplotlib.pyplot as plt
from src.diffusion.schedule import cosine_schedule
from src.diffusion.ddpm import q_sample


transform = T.ToTensor()
dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
img, label = dataset[0]

x0 = img.unsqueeze(0)*2 -1 

_,_,alpha_bar = cosine_schedule(1000)


time_steps =  [0,249,499,749,999]
fig,axes = plt.subplots(1,len(time_steps),figsize =(15,3))

for ax, t_val in zip(axes,time_steps):
    t = torch.tensor([t_val])
    xt,_ = q_sample(x0,t,alpha_bar)
    img = ((xt[0]+1)/2).clamp(0,1)
    img =  img.permute(1,2,0)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f'steps = {t_val}')

plt.tight_layout()
plt.savefig("assets/forward_process.png")