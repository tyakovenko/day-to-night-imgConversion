# Evaluation Results

**Eval pair:** `night.jpg` (input) / `day.jpg` (ground truth)  
**Date:** 2026-03-15  
**Metric:** channel-wise MSE in float64, pixel values in [0, 1]  
**Baseline:** `night.jpg` loaded as float32 RGB — same pipeline as model input — compared directly to `day.jpg` (no enhancement)  

---

## Results

| Model | MSE_R | MSE_G | MSE_B | MSE_avg |
|-------|------:|------:|------:|--------:|
| **Baseline (night vs day, no model)** | 0.113803 | 0.146743 | 0.156254 | **0.138933** |
| v1 (baseline U-Net) | 0.037988 | 0.035524 | 0.043262 | **0.038925** |
| v1-extended (+LOL data) | 0.050318 | 0.038497 | 0.040582 | **0.043132** |
| v2 (L1 + MS-SSIM loss) | 0.051711 | 0.038521 | 0.042594 | **0.044275** |
| v3 (residual + ColorLoss) | 0.048779 | 0.039017 | 0.050610 | **0.046135** |
| v4 (lamp suppression + GlobalContext) | 0.059873 | 0.036027 | 0.047200 | **0.047700** |

---

## Notes

- **v3 / v4 checkpoints** do not store `residual=True` in saved args; the flag is forced at inference time.
- **v4** additionally uses `GlobalContextEncoder` (per-channel mean/std/p10 injected at bottleneck); auto-detected from checkpoint.
- Padding: images with H or W not divisible by 16 are reflect-padded (bottom/right) before the U-Net and cropped back before MSE — the padded region never contributes to the score.
- All MSE values are in the [0, 1] pixel scale (not 0–255).
