import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim import AdamW
import copy
import os

from src.diffusion.schedule import cosine_schedule
from src.diffusion.ddpm import Gaussian_diff
from src.model.U_net import U_Net

# ---------- EMA ----------
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

# ---------- Checkpoint ----------
def save_checkpoint(path, epoch, model, optimizer, ema):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.shadow.state_dict(),
    }, path)

def load_checkpoint(path, model, optimizer, ema, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    ema.shadow.load_state_dict(ckpt['ema'])
    return ckpt['epoch']

# ---------- 训练 ----------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

beta, alpha, alpha_bar = cosine_schedule(T=1000)
diffusion = Gaussian_diff(alpha_bar)
net = U_Net(128, 512).to(device)
optim = AdamW(net.parameters(), lr=2e-4)
ema = EMA(net, decay=0.9999)

ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

start_epoch = 0
ckpt_path = os.path.join(ckpt_dir, "latest.pt")
if os.path.exists(ckpt_path):
    start_epoch = load_checkpoint(ckpt_path, net, optim, ema, device) + 1
    print(f"Resumed from epoch {start_epoch}")

train_epoch = 20000
for i in range(start_epoch, train_epoch):
    for images, labels in dataloader:
        images = images.to(device)
        loss = diffusion.compute_loss(net, images)
        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update(net)

    if i< 200 or i % 500 == 499:
        print(f"epoch {i+1} : loss = {loss:.4f}")
        save_checkpoint(ckpt_path, i, net, optim, ema)