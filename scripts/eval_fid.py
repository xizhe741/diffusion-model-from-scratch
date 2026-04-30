"""
=============================================================================
 FID evaluation for diffusion_model_from_scratch (DDPM, eps-prediction)
=============================================================================
最小评估流程: 单一配置 (默认 DDPM 反向采样, T=1000), 计算 FID 并打印.
不落盘, 直接喂 uint8 tensor 到 torchmetrics.image.fid.FrechetInceptionDistance.

依赖:
    pip install 'torchmetrics[image]'
    # 国内: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'torchmetrics[image]'

第一次运行需要下 InceptionV3 权重 (~92 MB) 到
    ~/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth
若已为 flow_matching_from_scratch 项目下过, 可直接复用 (同一个文件名).

Usage:
    python scripts/eval_fid.py --ckpt checkpoints/latest.pt
    python scripts/eval_fid.py --n 10000 --n-real 10000 --batch 128 --device cuda:1
=============================================================================
"""

import argparse
import os
import sys

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diffusion.schedule import cosine_schedule
from src.model.U_net import U_Net


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> [0, 255] uint8, 形状不变."""
    x = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
    return (x * 255.0).round().to(torch.uint8)


@torch.no_grad()
def ddpm_sample(net, beta, alpha, alpha_bar, n, device, img_size=32, channels=3):
    """与 scripts/sample_ckpt.py 中的 ddpm_sample 同接口, 移除控制台噪声.

    标准 DDPM 反向: x_{t-1} = (1/sqrt(alpha_t)) (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps)
                              + sigma_t * z,    sigma_t = sqrt(beta_t),  最后一步 z=0.
    """
    T_steps = len(beta)
    x = torch.randn(n, channels, img_size, img_size, device=device)
    for t in reversed(range(T_steps)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        eps = net(x, t_batch)
        a_t = alpha[t]
        ab_t = alpha_bar[t]
        coef = (1.0 - a_t) / torch.sqrt(1.0 - ab_t)
        mean = (1.0 / torch.sqrt(a_t)) * (x - coef * eps)
        if t > 0:
            sigma = torch.sqrt(beta[t])
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean
    return x


@torch.no_grad()
def collect_real(fid: FrechetInceptionDistance, batch_size: int, device, n_real):
    """喂 CIFAR-10 train 真实图. n_real=None 表示喂完整 50000 张."""
    transform = T.ToTensor()
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    seen = 0
    for images, _ in loader:
        images = images.to(device)
        images_u8 = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        fid.update(images_u8, real=True)
        seen += images.shape[0]
        if n_real is not None and seen >= n_real:
            break
    print(f"[real] fed {seen} CIFAR-10 train images to FID")


@torch.no_grad()
def collect_fake(fid, net, beta, alpha, alpha_bar, n_fake, batch_size, device):
    """采样 n_fake 张生成图喂 FID."""
    generated = 0
    while generated < n_fake:
        b = min(batch_size, n_fake - generated)
        x = ddpm_sample(net, beta, alpha, alpha_bar, b, device)
        fid.update(to_uint8(x), real=False)
        generated += b
        if generated % (batch_size * 10) == 0 or generated >= n_fake:
            print(f"[fake] sampled {generated}/{n_fake}")
    print(f"[fake] fed {generated} generated images to FID")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--n", type=int, default=10000, help="生成图张数")
    parser.add_argument("--n-real", type=int, default=10000,
                        help="参考真实图张数 (默认 10000, 与 --n 对称)")
    parser.add_argument("--batch", type=int, default=128, help="采样 batch")
    parser.add_argument("--T", type=int, default=1000, help="DDPM 反向步数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="如 'cuda:0' / 'cuda:1'; 默认自动选 cuda:0 或 cpu")
    parser.add_argument("--use-model", action="store_true",
                        help="用原模型权重而非 EMA (默认 EMA)")
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"loaded {args.ckpt}, epoch = {ckpt.get('epoch')}, keys = {list(ckpt.keys())}")

    net = U_Net(128, 512).to(device)
    state = ckpt["model"] if args.use_model else ckpt["ema"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (showing 3): {missing[:3]}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (showing 3): {unexpected[:3]}")
    net.eval()
    print(f"using {'raw model' if args.use_model else 'EMA'} weights")

    beta, alpha, alpha_bar = cosine_schedule(T=args.T)
    beta, alpha, alpha_bar = beta.to(device), alpha.to(device), alpha_bar.to(device)
    print(f"sampler: DDPM cosine schedule, T = {args.T}")

    torch.manual_seed(args.seed)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    collect_real(fid, batch_size=args.batch, device=device, n_real=args.n_real)
    collect_fake(fid, net, beta, alpha, alpha_bar,
                 n_fake=args.n, batch_size=args.batch, device=device)

    score = fid.compute().item()
    print(f"\nFID = {score:.4f}")
    print(f"  config: DDPM cosine, T={args.T}, "
          f"n_fake={args.n}, n_real={args.n_real}")


if __name__ == "__main__":
    main()
