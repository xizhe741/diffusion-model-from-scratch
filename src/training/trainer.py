import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim import AdamW
import copy
import os

from accelerate import Accelerator

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
accelerator = Accelerator()
device = accelerator.device
print(
    f"[rank={accelerator.process_index}/{accelerator.num_processes}] "
    f"local_rank={accelerator.local_process_index} "
    f"device={device} "
    f"LOCAL_RANK_env={os.environ.get('LOCAL_RANK')} "
    f"RANK_env={os.environ.get('RANK')} "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"torch.cuda.device_count()={torch.cuda.device_count()}",
    flush=True,
)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=384,
    shuffle=True,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
)

beta, alpha, alpha_bar = cosine_schedule(T=1000)
diffusion = Gaussian_diff(alpha_bar)
net = U_Net(128, 512)
optim = AdamW(net.parameters(), lr=2e-4)
ema = EMA(net, decay=0.9999)
ema.shadow.to(device)

net, optim, dataloader = accelerator.prepare(net, optim, dataloader)

ckpt_dir = "checkpoints"
if accelerator.is_main_process:
    os.makedirs(ckpt_dir, exist_ok=True)

start_epoch = 0
ckpt_path = os.path.join(ckpt_dir, "latest.pt")
# if os.path.exists(ckpt_path):
#     start_epoch = load_checkpoint(ckpt_path, accelerator.unwrap_model(net), optim, ema, device) + 1
#     if accelerator.is_main_process:
#         print(f"Resumed from epoch {start_epoch}")

train_epoch = 300
for i in range(start_epoch, train_epoch):
    for images, labels in dataloader:
        loss = diffusion.compute_loss(net, images)
        optim.zero_grad()
        accelerator.backward(loss)
        optim.step()
        ema.update(accelerator.unwrap_model(net))

    if accelerator.is_main_process and (i % 10 == 9 or i == train_epoch - 1):
        print(f"epoch {i+1} : loss = {loss:.4f}")
        save_checkpoint(ckpt_path, i, accelerator.unwrap_model(net), optim, ema)
        epoch_ckpt = os.path.join(ckpt_dir, f"epoch_{i+1:04d}.pt")
        save_checkpoint(epoch_ckpt, i, accelerator.unwrap_model(net), optim, ema)
