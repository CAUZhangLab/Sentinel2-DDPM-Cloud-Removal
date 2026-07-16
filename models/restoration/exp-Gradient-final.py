#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Model definitions
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.blk = nn.Sequential(LayerNorm2d(c), nn.Conv2d(c, c * 2, 1), nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2))
        self.conv_out = nn.Conv2d(c, c, 1)
        self.ffn = nn.Sequential(LayerNorm2d(c), nn.Conv2d(c, c * 2, 1), nn.Conv2d(c, c, 1))

    def forward(self, x):
        res = self.blk(x);
        x1, x2 = res.chunk(2, 1);
        x = x + self.conv_out(x1 * x2)
        identity = x;
        x = self.ffn[0](x);
        x = self.ffn[1](x);
        x1, x2 = x.chunk(2, 1);
        x = self.ffn[2](x1 * x2)
        return identity + x


class NAFNet(nn.Module):
    def __init__(self, in_c=6, width=32):
        super().__init__()
        self.intro = nn.Conv2d(in_c, width, 3, 1, 1);
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(12)])
        self.ending = nn.Conv2d(width, in_c, 3, 1, 1)

    def forward(self, x):
        res = x;
        x = self.intro(x);
        x = self.body(x);
        return self.ending(x) + res


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads, self.scale = num_heads, nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1);
        self.project = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape;
        q, k, v = self.qkv(x).chunk(3, 1)
        q = q.view(B, self.num_heads, -1, H * W);
        k = k.view(B, self.num_heads, -1, H * W);
        v = v.view(B, self.num_heads, -1, H * W)
        q = F.normalize(q, dim=-1);
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        return self.project((attn.softmax(-1) @ v).view(B, C, H, W))


class RestormerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim);
        self.attn = Attention(dim, num_heads);
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2),
                                 nn.GELU(), nn.Conv2d(dim * 2, dim, 1))

    def forward(self, x):
        x = x + self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class Restormer(nn.Module):
    def __init__(self, in_c=6, dim=48):
        super().__init__()
        self.embed = nn.Conv2d(in_c, dim, 3, 1, 1);
        self.encoder = nn.ModuleList([RestormerBlock(dim, 1) for _ in range(4)])
        self.final = nn.Conv2d(dim, in_c, 3, 1, 1)

    def forward(self, x):
        res = x;
        x = self.embed(x)
        for blk in self.encoder: x = blk(x)
        return self.final(x) + res


# 2. Dataset
class GradientTrainDataset(Dataset):
    def __init__(self, real_root, syn_root, num_real=5000, num_syn=500):
        self.samples = []
        # A. 真实训练数据: patches/train/cloudy -> patches/train/gt
        real_root = Path(real_root).resolve()
        real_cloudy = sorted(list((real_root / "cloudy").glob("*_cloudy.npy")))[:num_real]
        real_added = 0
        for cf in real_cloudy:
            gf = real_root / "gt" / cf.name.replace("_cloudy.npy", "_gt.npy")
            if gf.exists():
                self.samples.append((cf, gf))
                real_added += 1

        # B. 生成训练数据: 平铺在 exp3_pure_syn_5k 下
        syn_root = Path(syn_root).resolve()
        syn_cloudy = sorted(list(syn_root.glob("*_cloudy.npy")))[:num_syn]
        syn_added = 0
        for sf in syn_cloudy:
            sg = syn_root / sf.name.replace("_cloudy.npy", "_gt.npy")
            if sg.exists():
                self.samples.append((sf, sg))
                syn_added += 1
        print(f"✅ 混合完成: 真实({real_added}) + 生成({syn_added}) = {len(self.samples)} 组")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cf, gf = self.samples[idx]
        c, g = np.load(cf).astype(np.float32), np.load(gf).astype(np.float32)
        if c.shape[0] != 6: c, g = c.transpose(2, 0, 1), g.transpose(2, 0, 1)

        # 核心修正：解决 RuntimeError: Trying to resize storage
        c_t, g_t = torch.from_numpy(c), torch.from_numpy(g)
        if c_t.shape[-2:] != (256, 256):
            c_t = F.interpolate(c_t.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
            g_t = F.interpolate(g_t.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        return c_t, g_t


class PatchTestDataset(Dataset):

    def __init__(self, root, max_samples=1000):
        self.root = Path(root).resolve()
        cloudy_files = sorted(list((self.root / "cloudy").glob("*_cloudy.npy")))[:max_samples]
        self.samples = []
        for cf in cloudy_files:
            gf = self.root / "gt" / cf.name.replace("_cloudy.npy", "_gt.npy")
            if gf.exists(): self.samples.append((cf, gf))
        print(f"✅ 1000份验证集就绪")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cf, gf = self.samples[idx]
        c, g = np.load(cf).astype(np.float32), np.load(gf).astype(np.float32)
        if c.shape[0] != 6: c, g = c.transpose(2, 0, 1), g.transpose(2, 0, 1)
        c_t, g_t = torch.from_numpy(c), torch.from_numpy(g)
        if c_t.shape[-2:] != (256, 256):
            c_t = F.interpolate(c_t.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
            g_t = F.interpolate(g_t.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        return c_t, g_t


class SubsetPairsDataset(Dataset):
    """A lightweight dataset wrapper over a list of (cloudy_path, gt_path) pairs."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cf, gf = self.pairs[idx]
        c, g = np.load(cf).astype(np.float32), np.load(gf).astype(np.float32)
        if c.shape[0] != 6:
            c, g = c.transpose(2, 0, 1), g.transpose(2, 0, 1)
        c_t, g_t = torch.from_numpy(c), torch.from_numpy(g)
        if c_t.shape[-2:] != (256, 256):
            c_t = F.interpolate(c_t.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
            g_t = F.interpolate(g_t.unsqueeze(0), size=(256, 256), mode='bilinear').squeeze(0)
        return c_t, g_t


def evaluate(model, loader):
    model.eval()
    ps_l, ss_l = [], []
    with torch.no_grad():
        for c, g in loader:
            c, g = c.to(DEVICE), g.to(DEVICE)
            pred = model(c).clamp(0, 1)
            pn = pred.cpu().numpy().transpose(0, 2, 3, 1)
            gn = g.cpu().numpy().transpose(0, 2, 3, 1)
            for i in range(pn.shape[0]):
                ps_l.append(peak_signal_noise_ratio(gn[i], pn[i], data_range=1.0))
                ss_l.append(structural_similarity(gn[i], pn[i], channel_axis=-1, data_range=1.0))
    return float(np.mean(ps_l)), float(np.mean(ss_l))


# 3. Main Logic
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['nafnet', 'restormer'], required=True)
    parser.add_argument('--num-real', type=int, default=5000, help='Number of real samples to include')
    parser.add_argument('--num-syn', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--eval-test', action='store_true', help='Evaluate on patch test set after training')
    parser.add_argument('--test-max-samples', type=int, default=1000)
    parser.add_argument('--real-root', type=str, default="./data/real/train")
    parser.add_argument('--syn-root', type=str, default="./data/synthetic/train")
    parser.add_argument('--out-root', type=str, default="./training_results_gradient")
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--test-root', type=str, default="./data/test")
    parser.add_argument('--eval-only', action='store_true', help='Skip training and only evaluate the saved best checkpoint on test set')
    args = parser.parse_args()

    real_train_path = args.real_root
    syn_train_path = args.syn_root

    # Build mixed pool first, then split 90/10 (train/val) inside it
    random.seed(args.seed)
    model = NAFNet(in_c=6).to(DEVICE) if args.model == 'nafnet' else Restormer(in_c=6).to(DEVICE)

    run_name = args.run_name or f"Gradient_{args.model}_R{args.num_real}_S{args.num_syn}"
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        best_weight = out_dir / "best_model.pth"
        if not best_weight.exists():
            raise FileNotFoundError(f"Missing checkpoint for eval-only mode: {best_weight}")
        if not args.eval_test:
            raise ValueError("--eval-only requires --eval-test")
        test_loader = DataLoader(PatchTestDataset(args.test_root, max_samples=args.test_max_samples),
                                 batch_size=4, shuffle=False, num_workers=args.num_workers)
        model.load_state_dict(torch.load(best_weight, map_location=DEVICE))
        test_p, test_s = evaluate(model, test_loader)
        print(f"\n📌 TEST PSNR: {test_p:.2f}dB | TEST SSIM: {test_s:.4f} (samples={args.test_max_samples})")
        (out_dir / "metrics_test.txt").write_text(
            f"Model: {args.model}\nnum_real: {args.num_real}\nnum_syn: {args.num_syn}\nTEST_PSNR: {test_p:.4f}\nTEST_SSIM: {test_s:.4f}\nTime: {datetime.now()}\n"
        )
        return

    full_ds = GradientTrainDataset(real_train_path, syn_train_path, num_real=args.num_real, num_syn=args.num_syn)
    pairs = list(full_ds.samples)
    random.shuffle(pairs)
    val_n = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:val_n]
    train_pairs = pairs[val_n:]
    print(f"✅ Split: train={len(train_pairs)} val={len(val_pairs)} (val_ratio={args.val_ratio}, seed={args.seed})")

    train_loader = DataLoader(SubsetPairsDataset(train_pairs),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(SubsetPairsDataset(val_pairs),
                            batch_size=4, shuffle=False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    best_psnr, best_ssim = 0, 0
    best_epoch = -1
    for ep in range(args.epochs):
        model.train()
        for c, g in tqdm(train_loader, desc=f"Epoch {ep + 1}"):
            c, g = c.to(DEVICE), g.to(DEVICE);
            optimizer.zero_grad()
            l = criterion(model(c), g);
            l.backward();
            optimizer.step()

        # Validate on 10% split (NOT using test set)
        avg_p, avg_s = evaluate(model, val_loader)
        print(f"📊 VAL PSNR: {avg_p:.2f}dB | VAL SSIM: {avg_s:.4f}")

        if avg_p > best_psnr:
            best_psnr, best_ssim = avg_p, avg_s
            best_epoch = ep + 1
            torch.save(model.state_dict(), out_dir / "best_model.pth")

            # Metrics files (Reference: verify_patch_test_1000.py style)
            txt_content = (
                f"Model: {args.model}\n"
                f"num_real: {args.num_real}\n"
                f"Split: train={len(train_pairs)} val={len(val_pairs)} (val_ratio={args.val_ratio}, seed={args.seed})\n"
                f"BestEpoch: {best_epoch}\n"
                f"VAL_PSNR: {best_psnr:.4f}\n"
                f"VAL_SSIM: {best_ssim:.4f}\n"
                f"Time: {datetime.now()}"
            )
            (out_dir / "metrics.txt").write_text(txt_content)

            csv_content = (
                "model,num_real,num_syn,train_n,val_n,val_ratio,seed,best_epoch,val_psnr,val_ssim\n"
                f"{args.model},{args.num_real},{args.num_syn},{len(train_pairs)},{len(val_pairs)},{args.val_ratio},{args.seed},{best_epoch},{best_psnr:.4f},{best_ssim:.4f}"
            )
            (out_dir / "metrics.csv").write_text(csv_content)
            print(f"✨ 发现更好的结果并已更新相关文件")

    # Optional: evaluate on test once, using best weights
    if args.eval_test:
        test_loader = DataLoader(PatchTestDataset(args.test_root, max_samples=args.test_max_samples),
                                 batch_size=4, shuffle=False, num_workers=args.num_workers)
        best_weight = out_dir / "best_model.pth"
        if best_weight.exists():
            model.load_state_dict(torch.load(best_weight, map_location=DEVICE))
        test_p, test_s = evaluate(model, test_loader)
        print(f"\n📌 TEST PSNR: {test_p:.2f}dB | TEST SSIM: {test_s:.4f} (samples={args.test_max_samples})")
        (out_dir / "metrics_test.txt").write_text(
            f"Model: {args.model}\nnum_real: {args.num_real}\nnum_syn: {args.num_syn}\nTEST_PSNR: {test_p:.4f}\nTEST_SSIM: {test_s:.4f}\nTime: {datetime.now()}\n"
        )


if __name__ == "__main__": main()
