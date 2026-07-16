# Sentinel-2 DDPM Cloud Removal

Research-oriented PyTorch implementations for Sentinel-2 cloud-mask generation and multispectral cloud removal. The repository combines a denoising diffusion probabilistic model (DDPM) for learning binary cloud-mask distributions with compact NAFNet- and Restormer-style restoration models for reconstructing cloud-free multispectral imagery.

## Scope

This repository contains model and experiment code only. It does **not** include satellite imagery, preprocessed datasets, pretrained weights, checkpoints, credentials, or data-download utilities.

The code supports two related tasks:

1. **Cloud-mask diffusion**: train and evaluate a conditional-free DDPM that models single-channel binary cloud masks.
2. **Multispectral restoration**: train compact NAFNet- or Restormer-style networks on paired cloudy and cloud-free Sentinel-2 patches, including experiments that mix real and synthetic training pairs.

## Repository Structure

```text
.
├── .gitignore              # Data, credentials, checkpoints, caches, and outputs
├── environment.yml         # Reference Conda environment (Python 3.10)
├── pyproject.toml          # Project metadata, dependencies, and development tools
├── requirements.txt        # Pip-compatible runtime dependencies
└── models/
    ├── diffusion/
    │   ├── ddpm_train.py       # DDPM architecture, training loop, EMA, and checkpointing
    │   └── ddpm_evaluate.py    # Reconstruction metrics and cloud-mask sampling
    └── restoration/
        ├── exp-NAFNet.py       # NAFNet-style restoration baseline
        ├── exp-Restormer.py    # Restormer-style restoration baseline
        └── exp-Gradient-final.py
                                  # Mixed real/synthetic restoration experiment
```

## Requirements

- Python 3.9 or later
- PyTorch
- NumPy
- tqdm
- scikit-image
- torchvision (optional in the code, but included in the reference environments to enable PNG sampling)

### Pip

Create an isolated environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The equivalent metadata-driven installation is:

```bash
python -m pip install .
```

Install the optional development tools with:

```bash
python -m pip install ".[dev]"
```

### Conda

Create the reference Python 3.10 environment with:

```bash
conda env create -f environment.yml
conda activate sentinel2-ddpm-cloud-removal
```

Install a PyTorch build compatible with your operating system and CUDA version when GPU acceleration is required. All scripts fall back to CPU execution, but diffusion training is substantially faster on a CUDA-capable GPU.

## Data Conventions

### Cloud masks for DDPM training

The diffusion scripts expect the following layout:

```text
data/masks/
├── train/mask/*.npy
├── val/mask/*.npy
└── test/mask/*.npy
```

Each mask may have shape `(H, W)`, `(H, W, 1)`, or `(1, H, W)`. Values are thresholded at `0.5` when necessary, resized with nearest-neighbor interpolation, and normalized from `{0, 1}` to `{-1, 1}` before diffusion training.

### Paired restoration patches

The standalone NAFNet and Restormer scripts expect paired files in one directory:

```text
data/restoration/train/
├── <sample_id>_cloudy.npy
└── <sample_id>_gt.npy
```

The mixed real/synthetic experiment expects:

```text
data/
├── real/train/
│   ├── cloudy/*_cloudy.npy
│   └── gt/*_gt.npy
├── synthetic/train/
│   ├── *_cloudy.npy
│   └── *_gt.npy
└── test/
    ├── cloudy/*_cloudy.npy
    └── gt/*_gt.npy
```

Restoration arrays should be `float32` values scaled to `[0, 1]`. Both channel-first and supported channel-last arrays are accepted by the loaders. The default six-channel configuration is:

```text
[B02, B03, B04, B08, B11, B12]
```

For RGB-only experiments, use channel indices `2,1,0`, corresponding to `B04, B03, B02` in the internal band order.

## Usage

Run all commands from the repository root.

### Train the cloud-mask DDPM

```bash
python models/diffusion/ddpm_train.py \
  --data-root data/masks \
  --epochs 100 \
  --batch-size 8 \
  --image-size 256 \
  --ckpt-dir checkpoints/ddpm \
  --eval-test
```

The training script uses a time-conditioned U-Net, a linear beta schedule, optional spatial attention, exponential moving average (EMA) weights, and deterministic seeding. It writes the latest, best-validation, final model, and final EMA states to the checkpoint directory.

### Evaluate the DDPM and sample masks

```bash
python models/diffusion/ddpm_evaluate.py \
  --checkpoint checkpoints/ddpm/simple_ddpm_best.pth \
  --data-root data/masks \
  --split test \
  --sample-count 16 \
  --sample-output outputs/ddpm_samples \
  --metrics-json outputs/ddpm_metrics.json
```

Evaluation reports noise-prediction MSE, reconstructed-mask MSE and PSNR, Dice score, intersection over union (IoU), and cloud-coverage statistics. Sampling writes continuous and thresholded masks as NumPy arrays; a PNG grid is also written when torchvision is available.

### Train the NAFNet-style baseline

```bash
python models/restoration/exp-NAFNet.py \
  --data-dir data/restoration/train \
  --output-dir outputs/nafnet \
  --channels 0,1,2,3,4,5 \
  --epochs 100 \
  --batch-size 8
```

### Train the Restormer-style baseline

```bash
python models/restoration/exp-Restormer.py \
  --data-dir data/restoration/train \
  --output-dir outputs/restormer \
  --channels 0,1,2,3,4,5 \
  --epochs 100 \
  --batch-size 4
```

Both restoration baselines optimize an L1 reconstruction objective, report PSNR and SSIM on the validation split, use early stopping, and save the best model as `best_model.pth`.

### Train with mixed real and synthetic pairs

```bash
python models/restoration/exp-Gradient-final.py \
  --model nafnet \
  --num-real 5000 \
  --num-syn 500 \
  --real-root data/real/train \
  --syn-root data/synthetic/train \
  --test-root data/test \
  --out-root outputs/mixed_training \
  --epochs 100 \
  --eval-test
```

Set `--model` to either `nafnet` or `restormer`. The script combines the requested numbers of real and synthetic pairs, creates a seeded training/validation split, selects the best checkpoint by validation PSNR, and optionally evaluates it once on the held-out test set.

## Reproducibility

- All training scripts expose a random seed and initialize NumPy and PyTorch deterministically.
- Dataset splits should be recorded explicitly for comparable experiments. The standalone restoration scripts accept a JSON split file through `--split-json`.
- Report the selected bands, image size, split definition, random seed, checkpoint-selection rule, and hardware when publishing results.
- Exact floating-point results may vary across PyTorch, CUDA, and GPU versions.

## Metrics

- **DDPM**: noise-prediction MSE, reconstructed-mask MSE/PSNR, Dice, IoU, and cloud coverage.
- **Restoration**: PSNR and SSIM with an assumed data range of `[0, 1]`.

## Limitations

- The NAFNet and Restormer implementations are compact research baselines, not drop-in copies of the full official architectures.
- Dataset acquisition, preprocessing, synthetic cloudy-image construction, and geospatial validation are outside the scope of this repository.
- No pretrained weights or reference benchmark results are provided.
- The scripts have not been packaged as a stable Python library or production inference service.

## Security and Privacy

The repository intentionally excludes credentials, private filesystem paths, raw data, logs, caches, checkpoints, and intermediate experiment artifacts. Never commit access tokens, service passwords, or private dataset locations.

## License

No license file is currently included. Add an appropriate license before redistributing or reusing the code outside the permissions granted by the repository owner.
