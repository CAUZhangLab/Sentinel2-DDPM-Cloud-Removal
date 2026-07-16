#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, random, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# --- 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CHANNELS = [0, 1, 2, 3, 4, 5]  # 默认 6 通道（内部 npy: [B02,B03,B04,B08,B11,B12]）


def parse_channels(text: str):
    items = [t.strip() for t in text.split(",") if t.strip()]
    if not items:
        raise ValueError("channels must be like: 2,1,0")
    return [int(x) for x in items]


def set_global_seed(seed=42):
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, patience=30, delta=0.001):
        self.patience, self.delta = patience, delta
        self.counter, self.best_score, self.early_stop = 0, None, False

    def __call__(self, current_psnr):
        if self.best_score is None:
            self.best_score = current_psnr
        elif current_psnr < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score, self.counter = current_psnr, 0


class CloudRemovalDataset(Dataset):
    def __init__(self, data_root, *, channels, image_size: int = 256):
        self.data_root = Path(data_root).resolve()
        self.channels = list(channels)
        self.image_size = int(image_size)
        self.files = sorted(list(self.data_root.glob("*cloudy*.npy")))
        if not self.files: raise RuntimeError(f"❌ 路径无效: {self.data_root}")
        print(f"✅ NAFNet 加载样本: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        c_p = self.files[idx]
        g_p = Path(str(c_p).replace("cloudy", "gt"))
        c_d, g_d = np.load(c_p).astype(np.float32), np.load(g_p).astype(np.float32)
        if c_d.shape[0] > 13: c_d, g_d = c_d.transpose(2, 0, 1), g_d.transpose(2, 0, 1)
        c_t = torch.from_numpy(c_d[self.channels]).unsqueeze(0)
        g_t = torch.from_numpy(g_d[self.channels]).unsqueeze(0)
        if c_t.shape[-2:] != (self.image_size, self.image_size):
            c_t = F.interpolate(c_t, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            g_t = F.interpolate(g_t, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        return c_t.squeeze(0), g_t.squeeze(0)


# --- NAFNet 核心组件 ---
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True);
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * (x - u) / torch.sqrt(s + self.eps) + self.bias[:, None, None]


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        # 卷积层定义
        self.blk = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            nn.Conv2d(c * 2, c * 2, 3, 1, 1, groups=c * 2)
        )
        self.conv_out = nn.Conv2d(c, c, 1)
        self.ffn = nn.Sequential(
            LayerNorm2d(c),
            nn.Conv2d(c, c * 2, 1),
            nn.Conv2d(c, c, 1)
        )

    def forward(self, x):
        res = self.blk(x)
        x1, x2 = res.chunk(2, 1)
        x = x + self.conv_out(x1 * x2)
        identity = x
        x = self.ffn[0](x)  # LayerNorm
        x = self.ffn[1](x)  # Conv 1x1 (c -> 2c)
        x1, x2 = x.chunk(2, 1)  # SimpleGate
        x = self.ffn[2](x1 * x2)  # Conv 1x1 (c -> c)

        return identity + x


class NAFNet(nn.Module):
    def __init__(self, in_c=6, width=32):
        super().__init__()
        self.intro = nn.Conv2d(in_c, width, 3, 1, 1)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(12)])
        self.ending = nn.Conv2d(width, in_c, 3, 1, 1)

    def forward(self, x):
        res = x;
        x = self.intro(x);
        x = self.body(x);
        return self.ending(x) + res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='../training_results_5k')
    parser.add_argument('--run-name', type=str, default='', help='Optional fixed run directory name (no timestamp).')
    parser.add_argument('--channels', type=str, default="0,1,2,3,4,5",
                        help="Channel indices from internal npy. For RGB use 2,1,0 (B04,B03,B02).")
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-json', type=str, default='',
                        help='Optional split json with train/val sample ids (prefix without _cloudy.npy).')
    parser.add_argument('--eval-test', action='store_true', help='Evaluate on patch test split after training')
    parser.add_argument('--test-split-json', type=str, default='splits/patch_in_domain_v1.json')
    parser.add_argument('--patch-test-root', type=str, default='patch/patches/test')
    parser.add_argument('--test-max-samples', type=int, default=0)
    args = parser.parse_args()
    set_global_seed(args.seed)

    channels = parse_channels(args.channels)
    ds = CloudRemovalDataset(args.data_dir, channels=channels, image_size=args.image_size)
    if args.split_json:
        import json

        split = json.loads(Path(args.split_json).read_text(encoding='utf-8'))
        train_ids = split.get('train', [])
        val_ids = split.get('val', [])
        if not train_ids or not val_ids:
            raise ValueError(f"Invalid split json (need train/val): {args.split_json}")
        t_ds = CloudRemovalDataset(args.data_dir, channels=channels, image_size=args.image_size)
        t_ds.files = [Path(args.data_dir).resolve() / f"{sid}_cloudy.npy" for sid in train_ids]
        v_ds = CloudRemovalDataset(args.data_dir, channels=channels, image_size=args.image_size)
        v_ds.files = [Path(args.data_dir).resolve() / f"{sid}_cloudy.npy" for sid in val_ids]
    else:
        t_n = int(len(ds) * (1.0 - args.val_ratio))
        gen = torch.Generator().manual_seed(args.seed)
        t_ds, v_ds = torch.utils.data.random_split(ds, [t_n, len(ds) - t_n], generator=gen)

    t_ld = DataLoader(t_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    v_ld = DataLoader(v_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = NAFNet(in_c=len(channels)).to(DEVICE)
    opt, crit, early = optim.AdamW(model.parameters(), lr=args.lr), nn.L1Loss(), EarlyStopping(patience=30)

    run_name = args.run_name.strip() or f"NAFNet_{Path(args.data_dir).name}_{datetime.now().strftime('%m%d_%H%M')}"
    run_path = Path(args.output_dir) / run_name
    run_path.mkdir(parents=True, exist_ok=True);
    log_f = run_path / "metrics.csv"
    with open(log_f, 'w') as f:
        f.write("epoch,loss,psnr,ssim\n")

    print(f"🚀 开始 NAFNet 训练: {run_name} (epochs={args.epochs})")
    best_p = 0
    for ep in range(args.epochs):
        model.train();
        e_loss = 0
        pbar = tqdm(t_ld, desc=f"Epoch {ep + 1}/{args.epochs}")
        for c, g in pbar:
            c, g = c.to(DEVICE), g.to(DEVICE);
            opt.zero_grad();
            l = crit(model(c), g);
            l.backward();
            opt.step();
            e_loss += l.item()

        model.eval();
        ps, ss = [], []
        with torch.no_grad():
            for c, g in v_ld:
                c, g = c.to(DEVICE), g.to(DEVICE);
                p = model(c).clamp(0, 1)
                pn, gn = p.cpu().numpy().transpose(0, 2, 3, 1), g.cpu().numpy().transpose(0, 2, 3, 1)
                for i in range(pn.shape[0]):
                    ps.append(peak_signal_noise_ratio(gn[i], pn[i], data_range=1.0))
                    ss.append(structural_similarity(gn[i], pn[i], channel_axis=-1, data_range=1.0))

        avg_p, avg_s = np.mean(ps), np.mean(ss)
        print(f"📊 Val PSNR: {avg_p:.2f} | SSIM: {avg_s:.4f}")
        with open(log_f, 'a') as f:
            f.write(f"{ep + 1},{e_loss / len(t_ld)},{avg_p},{avg_s}\n")

        if avg_p > best_p:
            best_p = avg_p;
            torch.save(model.state_dict(), run_path / "best_model.pth")
            print(f"✨ 发现更好的 PSNR: {best_p:.2f}dB，已更新权重。")
        early(avg_p)
        if early.early_stop: print(f"🛑 触发时停。"); break

    # Optional: patch test evaluation
    if args.eval_test:
        import json

        split = json.loads(Path(args.test_split_json).read_text(encoding='utf-8'))
        ids = list(split['test'])
        if args.test_max_samples and args.test_max_samples > 0:
            ids = ids[:args.test_max_samples]

        class PatchTestDataset(Dataset):
            def __init__(self, root, sample_ids):
                self.root = Path(root).resolve()
                self.sample_ids = sample_ids
                self.cloudy_dir = self.root / 'cloudy'
                self.gt_dir = self.root / 'gt'

            def __len__(self):
                return len(self.sample_ids)

            def __getitem__(self, idx):
                sid = self.sample_ids[idx]
                c_p = self.cloudy_dir / f"{sid}_cloudy.npy"
                g_p = self.gt_dir / f"{sid}_gt.npy"
                c_d = np.load(c_p).astype(np.float32)
                g_d = np.load(g_p).astype(np.float32)
                if c_d.ndim == 3 and c_d.shape[0] not in (3, 4, 6, 10, 13) and c_d.shape[-1] in (3, 4, 6, 10, 13):
                    c_d = c_d.transpose(2, 0, 1)
                    g_d = g_d.transpose(2, 0, 1)
                c_t = torch.from_numpy(c_d[channels]).unsqueeze(0)
                g_t = torch.from_numpy(g_d[channels]).unsqueeze(0)
                if c_t.shape[-2:] != (args.image_size, args.image_size):
                    c_t = F.interpolate(c_t, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
                    g_t = F.interpolate(g_t, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
                return c_t.squeeze(0), g_t.squeeze(0)

        test_ds = PatchTestDataset(args.patch_test_root, ids)
        test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        best_weight = run_path / 'best_model.pth'
        if best_weight.exists():
            model.load_state_dict(torch.load(best_weight, map_location=DEVICE))

        model.eval()
        ps, ss = [], []
        with torch.no_grad():
            for c, g in tqdm(test_ld, desc='PatchTest'):
                c, g = c.to(DEVICE), g.to(DEVICE)
                p = model(c).clamp(0, 1)
                pn, gn = p.cpu().numpy().transpose(0, 2, 3, 1), g.cpu().numpy().transpose(0, 2, 3, 1)
                for i in range(pn.shape[0]):
                    ps.append(peak_signal_noise_ratio(gn[i], pn[i], data_range=1.0))
                    ss.append(structural_similarity(gn[i], pn[i], channel_axis=-1, data_range=1.0))

        test_p, test_s = float(np.mean(ps)), float(np.mean(ss))
        print(f"\n📌 TEST PSNR: {test_p:.2f}dB | TEST SSIM: {test_s:.4f} (n={len(test_ds)})")
        (run_path / 'metrics_test.txt').write_text(
            f"Model: nafnet\nTEST_PSNR: {test_p:.4f}\nTEST_SSIM: {test_s:.4f}\nTime: {datetime.now()}\n"
        )


if __name__ == "__main__": main()
