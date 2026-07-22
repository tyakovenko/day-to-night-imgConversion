---
title: Night to Day Enhancement
emoji: 🌙
sdk: docker
app_port: 7860
pinned: false
short_description: enhance low light images for day-like images
---

# Low-Light to Day Image Enhancement

A U-Net that brings low-light and night-time photos toward daylight appearance, evaluated by channel-wise MSE. Four model versions, each one built to fix a failure mode the last one showed.

- **Live demo:** [tyakovenko-night-to-day-enhancement.hf.space](https://tyakovenko-night-to-day-enhancement.hf.space)
- **Dataset:** [tyakovenko/night-to-day-enhancement](https://huggingface.co/datasets/tyakovenko/night-to-day-enhancement) (1,176 pairs, Transient Attributes)
- **Extended dataset:** `extended_manifest.csv` in this repo adds 420 LOL pairs
- **Models:** v1 ([`best.pt`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model)), v2 ([`best_v2.pt`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v2), L1 + MS-SSIM), v3 ([`best_v3.pt`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v3), residual + ColorLoss), v4 ([`best_v4.pt`](https://huggingface.co/tyakovenko/night-to-day-enhancement-model-v4), weighted L1 + global context)

## Model progression

The interesting part of this project is what the metrics don't say on their own. Raw MSE is lowest on v1, but each later version targets a specific defect (gray-washed outputs, blown-out lamps) that MSE alone rewards ignoring. See [`data/report.md`](data/report.md) for the per-version analysis.

| Version | Key change | Eval MSE | SSIM |
|---|---|---|---|
| v1 | Baseline MSE on Transient Attributes | **0.0389** | **0.533** |
| v1-extended | Fine-tuned on TA + LOL indoor data | 0.0431 | 0.465 |
| v2 | L1 + MS-SSIM instead of MSE | 0.0443 | 0.445 |
| v3 | Residual U-Net + ColorLoss (fixes gray outputs) | 0.0461 | 0.287 |
| v4 | Lamp-suppression losses + global context | 0.0477 | 0.310 |

v3 and v4 predict a residual, so pass `--residual` at inference.

## Run the demo (Docker)

No Python setup needed.

```bash
git clone https://github.com/tyakovenko/night-to-day-imgConversion.git
cd night-to-day-imgConversion
docker build -t night-to-day .          # CPU-only torch + deps in a venv
docker run -p 7860:7860 night-to-day    # UI at http://localhost:7860
```

On first launch the checkpoint (~22MB) downloads from HF Hub. Later runs use the Docker layer cache. Requires Docker; no GPU.

## Reproduce the experiment (Python)

```bash
git clone https://github.com/tyakovenko/night-to-day-imgConversion.git
cd night-to-day-imgConversion
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Training streams images from HF Hub on first run and caches them locally.

```bash
# v1 baseline (Transient Attributes, MSE)
python v1/train.py --epochs 30 --batch-size 8 --crop-size 128 \
  --base-filters 16 --lr 1e-4 --workers 4
# → checkpoints/best.pt at ~epoch 22, val MSE ~0.029, ~90-110s/epoch on CPU

# v4 (needs best.pt as warm start)
python v4/train_v4.py --epochs 30 --batch-size 4 --crop-size 176 --lr 1e-4 \
  --color-warmup-epochs 5 --global-context --augment-color \
  --init-checkpoint checkpoints/best.pt --workers 4
# → checkpoints/best_v4.pt at ~epoch 25, val MSE ~0.028, ~220s/epoch on CPU
```

Inference on a single image (add `--residual` for v3/v4):

```bash
python enhance.py --input night.jpg --checkpoint checkpoints/best.pt
python enhance.py --input night.jpg --checkpoint checkpoints/best_v4.pt --residual
python enhance.py --input night.jpg --reference day.jpg \
  --checkpoint checkpoints/best_v4.pt --residual   # with ground-truth eval
```

## Architecture

A 4-level U-Net (`base_filters=16`). Inputs are reflect-padded to the nearest multiple of 16, then cropped back, so any size works.

```
v1 / v1-extended / v2  (direct):
  Input → Encoder (4× ConvBlock + MaxPool) → Bottleneck → Decoder (4× Upsample + skip) → Conv1×1 + Sigmoid

v3  (residual):
  Input → Encoder → Bottleneck → Decoder → Conv1×1 + Tanh → delta
  Output = clamp(Input + delta, 0, 1)

v4  (residual + global context):
  Input → Encoder → Bottleneck ──┐
                                 ├─ add → Decoder → Conv1×1 + Tanh → delta
  GlobalContextEncoder(stats) ───┘
  GlobalContextEncoder: [mean/std/p10 per RGB channel] → Linear(9→32) → ReLU → Linear(32→bottleneck) → broadcast
```

## Repository structure

| File | Purpose |
|---|---|
| `model.py` | U-Net + `GlobalContextEncoder`; `residual` (v3+) and `use_global_context` (v4) flags |
| `losses.py` | `CombinedLoss`, `ColorLoss`, `WeightedL1Loss`, `LogL1Loss`, `V4Loss`, `PerceptualLoss` |
| `dataset.py` | Streams pairs from HF Hub via manifest CSV; gamma augmentation; global stats |
| `v1/`–`v4/` | Per-version training loops |
| `enhance.py` | Single-image inference + MSE eval |
| `app.py` | Gradio UI and Docker entry point (v4 default) |
| `Dockerfile` | Python 3.11-slim, venv, CPU-only torch |
| `low_light_manifest.csv` | 1,176 TA training pairs |
| `extended_manifest.csv` | 1,676 combined pairs (TA + LOL) |
| `data/report.md` | Full project report |
