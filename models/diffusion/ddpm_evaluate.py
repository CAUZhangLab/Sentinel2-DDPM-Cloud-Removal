import argparse
import json
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    from torchvision.utils import save_image
    _HAS_TORCHVISION = True
except ImportError:  # pragma: no cover - torchvision 可能未安装
    _HAS_TORCHVISION = False

from ddpm_train import (  # noqa: E402
    DiffusionUNet,
    EMAModel,
    SimpleDDPM,
    create_mask_dataloader,
    denormalize_mask,
    evaluate_ddpm,
    load_checkpoint,
    parse_int_tuple,
    parse_optional_int_tuple,
    seed_everything,
)


@torch.no_grad()
def compute_reconstruction_metrics(
    ddpm: SimpleDDPM,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    评估模型从任意时间步重建 x0 的能力，并计算常见指标。
    返回包含 MSE、PSNR、Dice、IoU 及掩模覆盖率等指标的字典。
    """
    ddpm.eval()
    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc="Reconstruction", unit="batch", leave=False)

    total_samples = 0
    mse_sum = 0.0
    psnr_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    coverage_true = 0.0
    coverage_pred = 0.0
    coverage_abs_diff = 0.0

    eps = 1e-8

    for batch in iterator:
        x0 = batch.to(device)
        bsz = x0.size(0)
        t = torch.randint(0, ddpm.T, (bsz,), device=device)
        xt, noise = ddpm.forward_diffuse(x0, t)
        pred_noise = ddpm(xt, t)

        # 预测的 x0（注意 clip 到 [-1,1]）
        alpha_bar_t = ddpm.alpha_bars.gather(0, t.long()).view(-1, 1, 1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
        pred_x0 = (xt - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        x0_denorm = denormalize_mask(x0).clamp(0.0, 1.0)
        pred_denorm = denormalize_mask(pred_x0).clamp(0.0, 1.0)

        mse = F.mse_loss(pred_denorm, x0_denorm, reduction="none").view(bsz, -1).mean(dim=1)
        mse_sum += mse.sum().item()

        psnr = 10.0 * torch.log10(torch.clamp(1.0 / (mse + eps), max=1e6))
        psnr_sum += psnr.sum().item()

        true_mask = (x0_denorm >= threshold).float()
        pred_mask = (pred_denorm >= threshold).float()

        inter = (true_mask * pred_mask).view(bsz, -1).sum(dim=1)
        true_sum = true_mask.view(bsz, -1).sum(dim=1)
        pred_sum = pred_mask.view(bsz, -1).sum(dim=1)
        union = true_sum + pred_sum - inter

        dice = (2 * inter + eps) / (true_sum + pred_sum + eps)
        iou = (inter + eps) / (union + eps)

        dice_sum += dice.sum().item()
        iou_sum += iou.sum().item()

        cov_true = x0_denorm.view(bsz, -1).mean(dim=1)
        cov_pred = pred_denorm.view(bsz, -1).mean(dim=1)
        coverage_true += cov_true.sum().item()
        coverage_pred += cov_pred.sum().item()
        coverage_abs_diff += torch.abs(cov_true - cov_pred).sum().item()

        total_samples += bsz

    if total_samples == 0:
        return {}

    return {
        "x0_mse": mse_sum / total_samples,
        "x0_psnr": psnr_sum / total_samples,
        "dice": dice_sum / total_samples,
        "iou": iou_sum / total_samples,
        "coverage_true": coverage_true / total_samples,
        "coverage_pred": coverage_pred / total_samples,
        "coverage_abs_diff": coverage_abs_diff / total_samples,
    }


@torch.no_grad()
def sample_masks(
    ddpm: SimpleDDPM,
    *,
    num_samples: int,
    image_size: int,
    device: torch.device,
    output_dir: Path,
    threshold: float = 0.5,
    use_tqdm: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Path]:
    """
    从训练好的 DDPM 采样生成掩模并保存 png/npy 文件。
    返回包含生成文件路径的字典。
    """
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    ddpm.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = torch.randn(num_samples, 1, image_size, image_size, device=device, generator=g)
    progress_iter = reversed(range(ddpm.T))
    if use_tqdm:
        progress_iter = tqdm(progress_iter, desc="Sampling", total=ddpm.T, leave=False)

    for t in progress_iter:
        t_batch = torch.tensor([t], device=device, dtype=torch.long).repeat(num_samples)
        beta_t = ddpm.betas[t_batch][:, None, None, None]
        alpha_t = ddpm.alphas[t_batch][:, None, None, None]
        alpha_bar_t = ddpm.alpha_bars[t_batch][:, None, None, None]
        if t > 0:
            alpha_bar_prev = ddpm.alpha_bars[t_batch - 1][:, None, None, None]
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)

        pred_noise = ddpm(samples, t_batch)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = coef1 * (samples - coef2 * pred_noise)

        if t > 0:
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            if g is not None:
                noise = torch.randn(samples.shape, device=device, generator=g)
            else:
                noise = torch.randn_like(samples)
            samples = mean + torch.sqrt(posterior_var) * noise
        else:
            samples = mean

    samples = samples.clamp(-1.0, 1.0)
    samples_denorm = denormalize_mask(samples).clamp(0.0, 1.0)
    samples_binary = (samples_denorm >= threshold).float()

    # 保存图像（若安装 torchvision）
    png_path = None
    if _HAS_TORCHVISION:
        png_path = output_dir / "samples.png"
        save_image(samples_denorm, png_path, nrow=min(4, num_samples))

    # 保存 npy
    npy_path = output_dir / "samples.npy"
    np.save(npy_path, samples_denorm.detach().cpu().numpy())

    # 保存二值化结果
    bin_path = output_dir / "samples_binary.npy"
    np.save(bin_path, samples_binary.detach().cpu().numpy())

    return {
        "png": png_path,
        "samples_npy": npy_path,
        "samples_binary_npy": bin_path,
    }


def summarize_metrics(tag: str, metrics: Dict[str, float]) -> None:
    print(f"\n[{tag}] 指标汇总:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.6f}")


def build_parser() -> argparse.ArgumentParser:
    default_data_root = Path(__file__).resolve().parents[1] / "patch" / "patches"
    parser = argparse.ArgumentParser(description="DDPM 训练后评估与采样脚本")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练生成的 checkpoint（*.pth）路径")
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="数据根目录（包含 train/val/test 子目录）")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="评估所用数据集 split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="评估与二值化阈值（默认 0.5）")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    parser.add_argument("--image-size", type=int, default=256, help="评估/采样统一的图像尺寸")

    parser.add_argument("--channel-mults", type=str, default="1,2,4", help="UNet 通道倍率，例如 1,2,4")
    parser.add_argument("--attn-levels", type=str, default="1", help="开启注意力的层索引，例如 1 或 0,1；传 none/空则关闭")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)

    parser.add_argument("--no-ema", action="store_true", help="不应用 EMA 权重进行评估")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="构造 EMA 模型时的 decay（仅用于加载状态）")
    parser.add_argument("--ema-fp32", action="store_true", help="以 FP32 维护 EMA shadow 参数")

    parser.add_argument("--sample-count", type=int, default=16, help="生成掩模的数量")
    parser.add_argument("--sample-output", type=Path, default=None, help="生成样本的保存目录（默认在 checkpoint 同级创建）")
    parser.add_argument("--sample-seed", type=int, default=None, help="采样使用的随机种子")
    parser.add_argument("--metrics-json", type=Path, default=None, help="将评估指标写入 JSON 文件的路径")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {args.checkpoint}")
    if not args.data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {args.data_root}")

    seed_everything(args.seed)

    device = torch.device(args.device) if args.device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"[INFO] 使用设备: {device}")

    pin_memory = device.type == "cuda"
    show_progress = not args.no_progress

    channel_mults = parse_int_tuple(args.channel_mults)
    attn_levels = parse_optional_int_tuple(args.attn_levels)
    if any(level >= len(channel_mults) for level in attn_levels):
        raise ValueError(f"attn_levels {attn_levels} 中的索引需小于 channel_multipliers 长度 {len(channel_mults)}")

    dataloader = create_mask_dataloader(
        args.data_root,
        args.split,
        batch_size=args.batch_size,
        size=args.image_size,
        augment=False,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )
    example_item = next(iter(dataloader))
    _, _, h, w = example_item.shape
    print(f"[INFO] {args.split} 集样本数: {len(dataloader.dataset)}, 解析到尺寸: {h}x{w}")

    model = DiffusionUNet(
        in_channels=1,
        base_channels=args.base_channels,
        channel_mults=channel_mults,
        time_dim=args.time_dim,
        attn_levels=attn_levels,
    )
    ddpm = SimpleDDPM(model, T=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)

    ema_model = None
    if not args.no_ema:
        ema_model = EMAModel(ddpm, decay=args.ema_decay, use_fp32_shadow=args.ema_fp32)

    checkpoint = load_checkpoint(args.checkpoint, ddpm, ema_model=ema_model, map_location="cpu")
    ddpm.to(device)
    if ema_model is not None:
        ema_model.to(device)

    # 评估 raw 权重
    raw_state = deepcopy(ddpm.state_dict())
    metrics: Dict[str, Dict[str, float]] = {}

    noise_mse_raw = evaluate_ddpm(ddpm, dataloader, device, ema_model=None)
    recon_metrics_raw = compute_reconstruction_metrics(
        ddpm,
        dataloader,
        device,
        threshold=args.threshold,
        show_progress=show_progress,
    )
    recon_metrics_raw["noise_mse"] = noise_mse_raw
    metrics["raw"] = recon_metrics_raw
    summarize_metrics("RAW", recon_metrics_raw)

    # 如果有 EMA，评估 EMA 权重
    if ema_model is not None and checkpoint.get("ema_state") is not None:
        ema_model.copy_to(ddpm)
        noise_mse_ema = evaluate_ddpm(ddpm, dataloader, device, ema_model=None)
        recon_metrics_ema = compute_reconstruction_metrics(
            ddpm,
            dataloader,
            device,
            threshold=args.threshold,
            show_progress=show_progress,
        )
        recon_metrics_ema["noise_mse"] = noise_mse_ema
        metrics["ema"] = recon_metrics_ema
        summarize_metrics("EMA", recon_metrics_ema)

        # 评估结束后恢复原始权重，方便后续需要同时比较 raw 与 ema
        ddpm.load_state_dict(raw_state)

    # 采样（优先使用 EMA 权重）
    sample_dir = args.sample_output or args.checkpoint.parent / "evaluation_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    if "ema" in metrics:
        ema_model.copy_to(ddpm)
        tag = "ema"
    else:
        tag = "raw"

    samples_paths = sample_masks(
        ddpm,
        num_samples=args.sample_count,
        image_size=args.image_size,
        device=device,
        output_dir=sample_dir / tag,
        threshold=args.threshold,
        use_tqdm=show_progress,
        seed=args.sample_seed,
    )
    print(f"[INFO] 已保存采样结果至 {samples_paths['samples_npy']}")
    if samples_paths["png"] is not None:
        print(f"[INFO] 样本可视化: {samples_paths['png']}")

    # 如果生成后需要恢复原始状态（供后续脚本使用）
    if "ema" in metrics:
        ddpm.load_state_dict(raw_state)

    # 输出汇总
    print("\n=== 综合评估结果 ===")
    for key, value in metrics.items():
        summarize_metrics(key.upper(), value)

    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 指标写入: {args.metrics_json}")


if __name__ == "__main__":
    main()
