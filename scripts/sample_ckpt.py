"""
Load checkpoints/latest.pt and run DDPM reverse sampling on CIFAR-10 sized images.
Saves an 8x8 grid PNG so you can eyeball whether the model learned anything.

Usage (run from project root):
    python scripts/sample_ckpt.py
    python scripts/sample_ckpt.py --ckpt checkpoints/latest.pt --n 64 --out samples.png
    python scripts/sample_ckpt.py --use-model   # use raw model weights instead of EMA
"""
import argparse
import os
import sys
import torch
from torchvision.utils import save_image, make_grid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.diffusion.schedule import cosine_schedule
from src.model.U_net import U_Net


@torch.no_grad()
def ddpm_sample(net, beta, alpha, alpha_bar, n, device, img_size=32, channels=3):
    T = len(beta)
    beta = beta.to(device)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)

    x = torch.randn(n, channels, img_size, img_size, device=device)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        eps = net(x, t_batch)

        a_t = alpha[t]
        ab_t = alpha_bar[t]
        coef = (1 - a_t) / torch.sqrt(1 - ab_t)
        mean = (1.0 / torch.sqrt(a_t)) * (x - coef * eps)

        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta[t])
            x = mean + sigma * noise
        else:
            x = mean

        if t % 100 == 0:
            print(f"  step t={t:4d}  mean={x.mean().item():+.3f}  std={x.std().item():.3f}", flush=True)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/latest.pt")
    parser.add_argument("--out", default="samples.png")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--use-model", action="store_true",
                        help="Use raw model weights instead of EMA (default: EMA)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    print(f"loaded {args.ckpt}, keys = {list(ckpt.keys())}, epoch = {ckpt.get('epoch')}")

    net = U_Net(128, 512).to(device)
    state = ckpt["model"] if args.use_model else ckpt["ema"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (showing 3): {missing[:3]}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (showing 3): {unexpected[:3]}")
    net.eval()
    print(f"using {'model' if args.use_model else 'ema'} weights")

    beta, alpha, alpha_bar = cosine_schedule(T=args.T)

    print(f"sampling {args.n} images over {args.T} steps...")
    x = ddpm_sample(net, beta, alpha, alpha_bar, args.n, device)

    x = (x.clamp(-1, 1) + 1) / 2
    grid = make_grid(x, nrow=8)
    save_image(grid, args.out)
    print(f"saved -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
