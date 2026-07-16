import argparse
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 数据集类（读取 mask npy，统一为 (1,H,W) Tensor，并可统一尺寸）
class CloudMaskDataset(Dataset):
    def __init__(self, mask_dir, size=256, augment=True, normalize=True):
        mask_dir = Path(mask_dir)
        self.mask_paths = sorted(str(p) for p in mask_dir.glob("*.npy"))
        if not self.mask_paths:
            raise FileNotFoundError(f"未在 {mask_dir} 找到 *.npy 掩模文件")
        self.size = size  # 统一到指定尺寸（设为 None 则保持原尺寸）
        self.augment = augment
        self.normalize = normalize

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        path = self.mask_paths[idx]
        mask = np.load(path).astype(np.float32)  # (H,W) 或 (H,W,1)
        # 统一为 (1,H,W)
        if mask.ndim == 2:
            mask = mask[None, ...]  # (1,H,W)
        elif mask.ndim == 3:
            # 支持 (H,W,1) 或 已经 (1,H,W)
            if mask.shape[0] == 1:
                pass  # (1,H,W)
            elif mask.shape[-1] == 1:
                mask = np.transpose(mask, (2, 0, 1))  # (H,W,1)->(1,H,W)
            else:
                raise ValueError(f"mask 维度非预期: {mask.shape} @ {path}")
        else:
            raise ValueError(f"mask 维度非预期: {mask.shape} @ {path}")

        # 二值规范化（确保 {0,1}）
        if not np.array_equal(np.unique(mask), np.array([0.0, 1.0])):
            mask = (mask >= 0.5).astype(np.float32)

        mask = torch.from_numpy(mask)  # (1,H,W)

        # 统一尺寸（混有 256 与 512，batch 需一致）
        if self.size is not None and (mask.shape[-2] != self.size or mask.shape[-1] != self.size):
            mask = F.interpolate(mask.unsqueeze(0), size=(self.size, self.size), mode='nearest').squeeze(0)

        # 简单增强
        if self.augment:
            if random.random() < 0.5:
                mask = torch.flip(mask, dims=[-1])  # 水平
            if random.random() < 0.5:
                mask = torch.flip(mask, dims=[-2])  # 垂直

        if self.normalize:
            mask = mask * 2.0 - 1.0  # {0,1} -> {-1,1}

        return mask  # (1, size, size)


def denormalize_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask + 1.0) * 0.5


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(seed: int):
    def init_fn(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return init_fn


def create_mask_dataloader(
    data_root: Path,
    split: str,
    batch_size: int,
    *,
    size: int = 256,
    augment: bool = False,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> DataLoader:
    mask_dir = Path(data_root) / split / "mask"
    dataset = CloudMaskDataset(mask_dir, size=size, augment=augment, normalize=normalize)
    worker_init = _worker_init_fn(seed) if (seed is not None and num_workers > 0) else None
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        generator=generator,
    )


class SinusoidalPositionEmbedding(nn.Module):
    """标准正弦时间嵌入，用于 DDPM（参考 Ho et al. 2020）."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32, device=device)
            / max(half_dim - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """带时间条件的残差块，使用 GroupNorm 和 SiLU 激活."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """空间自注意力块，缓解网络在粗尺度的表达瓶颈."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = self.norm(x)
        q = self.q(h_in).reshape(b, c, -1)
        k = self.k(h_in).reshape(b, c, -1)
        v = self.v(h_in).reshape(b, c, -1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2).reshape(b, c, h, w)
        return x + self.out(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class DiffusionUNet(nn.Module):
    """尺度一致的 UNet（遵循 DDPM 论文结构），强化多尺度表达与时间调制."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Iterable[int] = (1, 2, 4),
        time_dim: int = 256,
        attn_levels: Tuple[int, ...] = (1,),
    ):
        super().__init__()
        channel_mults = tuple(channel_mults)
        num_levels = len(channel_mults)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attn_levels = set(attn_levels)

        channels = base_channels

        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            res1 = ResidualBlock(channels, out_channels, time_dim)
            res2 = ResidualBlock(out_channels, out_channels, time_dim)
            attn = AttentionBlock(out_channels) if level in self.attn_levels else nn.Identity()
            downsample = Downsample(out_channels) if level < num_levels - 1 else nn.Identity()
            self.downs.append(nn.ModuleList([res1, res2, attn, downsample]))
            channels = out_channels

        self.mid_block1 = ResidualBlock(channels, channels, time_dim)
        self.mid_attn = AttentionBlock(channels)
        self.mid_block2 = ResidualBlock(channels, channels, time_dim)

        for level, mult in enumerate(reversed(tuple(channel_mults))):
            out_channels = base_channels * mult
            res1 = ResidualBlock(channels + out_channels, out_channels, time_dim)
            res2 = ResidualBlock(out_channels + out_channels, out_channels, time_dim)
            attn = AttentionBlock(out_channels) if (num_levels - 1 - level) in self.attn_levels else nn.Identity()
            upsample = Upsample(out_channels) if level < num_levels - 1 else nn.Identity()
            self.ups.append(nn.ModuleList([res1, res2, attn, upsample]))
            channels = out_channels

        self.final_norm = nn.GroupNorm(8, channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)
        skips = []

        for res1, res2, attn, down in self.downs:
            h = res1(h, t_emb)
            skips.append(h)
            h = res2(h, t_emb)
            skips.append(h)
            h = attn(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for res1, res2, attn, up in self.ups:
            h = torch.cat([h, skips.pop()], dim=1)
            h = res1(h, t_emb)
            h = torch.cat([h, skips.pop()], dim=1)
            h = res2(h, t_emb)
            h = attn(h)
            h = up(h)

        h = self.final_conv(self.final_act(self.final_norm(h)))
        return h


# DDPM类（用 register_buffer，索引用 gather）
class SimpleDDPM(nn.Module):
    def __init__(self, model, T=200, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, T, dtype=torch.float32))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

    def forward_diffuse(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars.gather(0, t.long()).view(-1, 1, 1, 1)  # 保证 long 索引
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_1m_ab = torch.sqrt(1.0 - alpha_bar_t)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

    def forward(self, xt, t):
        return self.model(xt, t)  # 预测噪声


class EMAModel:
    """
    Exponential Moving Average (EMA) for model parameters.

    - 使用 dict[str, Tensor] 存储 shadow 参数，键为参数名，便于与 named_parameters 对齐。
    - 严格对齐设备和 dtype，避免 "found at least two devices" 等错误。
    - 仅对 requires_grad=True 的参数做 EMA。
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, *, use_fp32_shadow: bool = False) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay: float = float(decay)
        self.use_fp32_shadow: bool = bool(use_fp32_shadow)

        # 以当前模型第一块参数的设备为准
        device = next(model.parameters()).device

        self.shadow: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                t = p.detach().clone()
                # 可选：将 shadow 始终用 fp32 存（提升数值稳定性）
                if self.use_fp32_shadow:
                    t = t.float()
                # 放到与模型一致的设备
                t = t.to(device)
                self.shadow[name] = t

        # 备份容器（用于 store/restore）
        self._backup: Dict[str, torch.Tensor] = {}

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "EMAModel":
        """将 EMA shadow 参数迁移到指定设备/数据类型。"""
        with torch.no_grad():
            for k, v in self.shadow.items():
                self.shadow[k] = v.to(device=device, dtype=(dtype or v.dtype))
        # 备份也一并迁移，便于 restore
        if self._backup:
            with torch.no_grad():
                for k, v in self._backup.items():
                    self._backup[k] = v.to(device=device, dtype=(dtype or v.dtype))
        return self

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        用当前模型参数更新 EMA：
            shadow = decay * shadow + (1 - decay) * param
        要求：model 与 shadow 在同一设备或可被自动对齐（本函数会对齐）。
        """
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # 如果遇到训练过程中新增的参数（罕见），初始化一份
            if name not in self.shadow:
                t = param.detach().clone()
                t = (t.float() if self.use_fp32_shadow else t)
                t = t.to(param.device)
                self.shadow[name] = t
                continue

            shadow_param = self.shadow[name]

            # 若设备 / dtype 不一致，做一次对齐到 param 的设备/dtype（数值稳定性优先）
            target_dtype = torch.float32 if self.use_fp32_shadow else param.dtype
            if shadow_param.device != param.device or shadow_param.dtype != target_dtype:
                shadow_param = shadow_param.to(device=param.device, dtype=target_dtype)
                self.shadow[name] = shadow_param  # 覆盖回字典

            # EMA 更新
            shadow_param.mul_(self.decay).add_(param.detach().to(dtype=target_dtype), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """
        将 EMA 的 shadow 参数拷贝到给定模型（常用于评估或保存 EMA 权重）。
        只覆盖 requires_grad=True 且存在于 shadow 的参数。
        """
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                # 对缺失条目保持原样
                continue
            src = self.shadow[name]
            # 若 dtype 不同，按模型参数 dtype 对齐
            if src.dtype != param.dtype or src.device != param.device:
                src = src.to(device=param.device, dtype=param.dtype)
            param.data.copy_(src.data)

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """
        备份当前模型参数（常在做临时评估前调用：store -> copy_to -> eval -> restore）。
        """
        self._backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self._backup[name] = param.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """恢复先前通过 store() 备份的模型参数。"""
        if not self._backup:
            return
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._backup:
                src = self._backup[name]
                if src.device != param.device or src.dtype != param.dtype:
                    src = src.to(device=param.device, dtype=param.dtype)
                param.data.copy_(src.data)
        self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        导出 EMA shadow 的状态字典。保持当前设备/精度，交给上层决定是否 .cpu() 存盘。
        """
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *, map_location: Optional[torch.device] = None) -> None:
        """
        加载 EMA shadow 的状态字典。可用 map_location 指定目标设备（例如 torch.device('cuda:4')）。
        如果键集合不同，会取交集加载，避免因结构变动崩溃。
        """
        new_shadow: Dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            t = tensor.detach().clone()
            if map_location is not None:
                t = t.to(map_location)
            # 若启用 fp32 shadow，确保是 float32
            if self.use_fp32_shadow and t.dtype != torch.float32:
                t = t.float()
            new_shadow[name] = t
        # 与现有 shadow 做对齐（保留旧有未出现在 state_dict 的条目）
        self.shadow.update(new_shadow)

def auto_select_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            free_list = []
            for idx in range(torch.cuda.device_count()):
                with torch.cuda.device(idx):
                    try:
                        free_bytes, _ = torch.cuda.mem_get_info()
                    except RuntimeError:
                        free_bytes = 0
                free_list.append((free_bytes, idx))
            best_idx = max(free_list, key=lambda x: x[0])[1]
            torch.cuda.set_device(best_idx)
            return torch.device(f"cuda:{best_idx}")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


# 训练函数
def evaluate_ddpm(
    ddpm: SimpleDDPM,
    dataloader: DataLoader,
    device: torch.device,
    *,
    ema_model: Optional[EMAModel] = None,
) -> float:
    ddpm.eval()
    if ema_model is not None:
        ema_model.store(ddpm)
        ema_model.copy_to(ddpm)

    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            t = torch.randint(0, ddpm.T, (batch.size(0),), device=device)
            xt, noise = ddpm.forward_diffuse(batch, t)
            per_elem = F.mse_loss(pred_noise := ddpm(xt, t), noise, reduction="none")
            per_sample = per_elem.view(per_elem.size(0), -1).mean(dim=1)
            total_loss += per_sample.sum().item()
            total_samples += batch.size(0)

    if ema_model is not None:
        ema_model.restore(ddpm)
    ddpm.train()
    return total_loss / max(1, total_samples)


def train_ddpm(
    ddpm: SimpleDDPM,
    train_loader: DataLoader,
    *,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 3,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
    ema_model: Optional[EMAModel] = None,
    ckpt_dir: Optional[Path] = None,
):
    if device is None:
        device = auto_select_device()
    ddpm.to(device)

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_fn = tqdm.write if show_progress else print
    log_fn(f"[INFO] 训练设备: {device}")
    log_fn(f"[INFO] 训练样本数: {len(train_loader.dataset)}, 每轮步数: {len(train_loader)}")
    if val_loader is not None:
        log_fn(f"[INFO] 验证样本数: {len(val_loader.dataset)}, 验证步数: {len(val_loader)}")

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)
    best_val = float("inf") if val_loader is not None else None
    best_ckpt_path = None
    last_ckpt_path = ckpt_dir / "simple_ddpm_last.pth"

    epoch_range = range(1, epochs + 1)
    if show_progress:
        epoch_range = tqdm(epoch_range, desc="Epochs", unit="epoch")

    for epoch in epoch_range:
        ddpm.train()
        running_loss = 0.0
        running_samples = 0
        batch_iter = train_loader
        if show_progress:
            batch_iter = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", unit="batch", leave=False)

        for batch in batch_iter:
            batch = batch.to(device)
            t = torch.randint(0, ddpm.T, (batch.size(0),), device=device)
            xt, noise = ddpm.forward_diffuse(batch, t)
            mse = F.mse_loss(pred_noise := ddpm(xt, t), noise, reduction="none")
            per_sample = mse.view(mse.size(0), -1).mean(dim=1)
            loss = per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update(ddpm)

            running_loss += per_sample.sum().item()
            running_samples += batch.size(0)

        train_loss = running_loss / max(1, running_samples)
        val_loss = None
        if val_loader is not None:
            val_loss = evaluate_ddpm(ddpm, val_loader, device, ema_model=ema_model)

        log_parts = [f"Epoch {epoch}/{epochs}", f"train_loss={train_loss:.6f}"]
        if val_loss is not None:
            log_parts.append(f"val_loss={val_loss:.6f}")
        log_fn(" | ".join(log_parts))

        state = {
            "epoch": epoch,
            "model_state": ddpm.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ema_state": ema_model.state_dict() if ema_model is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(state, last_ckpt_path)

        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            best_ckpt_path = ckpt_dir / "simple_ddpm_best.pth"
            torch.save(state, best_ckpt_path)

    final_model_path = ckpt_dir / "simple_ddpm_final.pth"
    torch.save(ddpm.state_dict(), final_model_path)
    final_ema_path = None
    if ema_model is not None:
        final_ema_path = ckpt_dir / "simple_ddpm_final_ema.pth"
        torch.save(ema_model.state_dict(), final_ema_path)

    log_fn(f"[INFO] 训练完成，最新权重保存在: {last_ckpt_path}")
    if best_ckpt_path is not None:
        log_fn(f"[INFO] 验证最佳权重保存在: {best_ckpt_path} (val_loss={best_val:.6f})")
    log_fn(f"[INFO] 模型 state_dict 另存为: {final_model_path}")
    if final_ema_path is not None:
        log_fn(f"[INFO] EMA state_dict 另存为: {final_ema_path}")

    return {
        "best_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
        "last_path": str(last_ckpt_path),
        "final_model_path": str(final_model_path),
        "final_ema_path": str(final_ema_path) if final_ema_path is not None else None,
        "best_val": best_val,
    }


def load_checkpoint(
    path: Path,
    ddpm: SimpleDDPM,
    *,
    ema_model: Optional[EMAModel] = None,
    map_location: Optional[torch.device] = None,
):
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    ddpm.load_state_dict(checkpoint["model_state"])
    if ema_model is not None and checkpoint.get("ema_state") is not None:
        ema_model.load_state_dict(checkpoint["ema_state"])
    return checkpoint


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise ValueError("需要至少提供一个整数")
    return tuple(int(item) for item in items)


def parse_optional_int_tuple(text: str) -> Tuple[int, ...]:
    text = text.strip()
    if not text or text.lower() in {"none", "null"}:
        return tuple()
    return parse_int_tuple(text)


def build_arg_parser() -> argparse.ArgumentParser:
    default_data_root = Path(__file__).resolve().parents[1] / "patch" / "patches"
    parser = argparse.ArgumentParser(description="云掩模 DDPM 训练脚本")
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="数据根目录，需包含 train/val/test 子目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--test-batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="关闭训练数据增强（翻转）")
    parser.add_argument("--no-val", action="store_true", help="禁用验证循环")
    parser.add_argument("--eval-test", action="store_true", help="训练结束后在测试集上评估")
    parser.add_argument("--no-progress", action="store_true", help="关闭 tqdm 进度条")
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="自定义 checkpoint 输出目录")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4", help="例如 1,2,4")
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--attn-levels", type=str, default="1", help="例如 1 或 0,1；传 none 关闭注意力")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true", help="禁用 EMA")
    parser.add_argument("--device", type=str, default=None, help="指定设备，例如 cuda:0 或 cpu")
    parser.add_argument("--no-pin-memory", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"数据根目录 {args.data_root} 不存在")

    seed_everything(args.seed)

    device = torch.device(args.device) if args.device is not None else auto_select_device()
    pin_memory = False if args.no_pin_memory else device.type == "cuda"

    channel_mults = parse_int_tuple(args.channel_mults)
    attn_levels = parse_optional_int_tuple(args.attn_levels)
    if any(level >= len(channel_mults) for level in attn_levels):
        raise ValueError(f"attn_levels {attn_levels} 中的索引需要小于 channel_mults 长度 {len(channel_mults)}")

    val_batch_size = args.val_batch_size or args.batch_size
    test_batch_size = args.test_batch_size or args.batch_size

    train_loader = create_mask_dataloader(
        args.data_root,
        "train",
        batch_size=args.batch_size,
        size=args.image_size,
        augment=not args.no_augment,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
    )

    val_loader = None
    if not args.no_val:
        val_mask_dir = args.data_root / "val" / "mask"
        if val_mask_dir.exists():
            val_loader = create_mask_dataloader(
                args.data_root,
                "val",
                batch_size=val_batch_size,
                size=args.image_size,
                augment=False,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                seed=args.seed,
            )
        else:
            print(f"[WARN] 未找到验证掩模目录: {val_mask_dir}，将跳过验证。")

    test_loader = None
    if args.eval_test:
        test_mask_dir = args.data_root / "test" / "mask"
        if test_mask_dir.exists():
            test_loader = create_mask_dataloader(
                args.data_root,
                "test",
                batch_size=test_batch_size,
                size=args.image_size,
                augment=False,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                seed=args.seed,
            )
        else:
            print(f"[WARN] 未找到测试掩模目录: {test_mask_dir}，将跳过测试评估。")

    model = DiffusionUNet(
        in_channels=1,
        base_channels=args.base_channels,
        channel_mults=channel_mults,
        time_dim=args.time_dim,
        attn_levels=attn_levels,
    )
    ddpm = SimpleDDPM(model, T=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)
    ema_model = None if args.no_ema else EMAModel(ddpm, decay=args.ema_decay)

    ckpt_info = train_ddpm(
        ddpm,
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        show_progress=not args.no_progress,
        ema_model=ema_model,
        ckpt_dir=args.ckpt_dir,
    )

    if args.eval_test and test_loader is not None:
        ckpt_path = ckpt_info["best_path"] or ckpt_info["last_path"]
        print(f"[INFO] 测试前加载 checkpoint: {ckpt_path}")
        load_checkpoint(Path(ckpt_path), ddpm, ema_model=ema_model, map_location=device)
        test_loss = evaluate_ddpm(ddpm, test_loader, device, ema_model=ema_model if not args.no_ema else None)
        print(f"[INFO] 测试集 MSE: {test_loss:.6f}")


if __name__ == "__main__":
    main()
