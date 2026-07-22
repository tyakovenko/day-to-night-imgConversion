"""
Compute initial (baseline) and per-model channel-wise MSE for the eval pair.

Baseline: night.jpg loaded the same way as model inputs (float32 RGB [0,1]),
           compared directly to day.jpg — no model involved.

Per-model: runs enhance() from enhance.py for each checkpoint and evaluates
           against day.jpg using the same channel_mse_eval logic.

Output: evals.md in the project root.

Usage:
    python compute_evals.py
    python compute_evals.py --night night.jpg --day day.jpg
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np

from enhance import load_image_rgb, enhance, channel_mse_eval


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    {
        "name": "v1 (baseline U-Net)",
        "checkpoint": "checkpoints/best.pt",
        "residual": False,
        "base_filters": 16,
    },
    {
        "name": "v1-extended (+LOL data)",
        "checkpoint": "checkpoints/best_extended.pt",
        "residual": False,
        "base_filters": 16,
    },
    {
        "name": "v2 (L1 + MS-SSIM loss)",
        "checkpoint": "checkpoints/best_v2.pt",
        "residual": False,
        "base_filters": 16,
    },
    {
        "name": "v3 (residual + ColorLoss)",
        "checkpoint": "checkpoints/best_v3.pt",
        "residual": True,   # not saved in checkpoint — must be forced
        "base_filters": 16,
    },
    {
        "name": "v4 (lamp suppression + GlobalContext)",
        "checkpoint": "checkpoints/best_v4.pt",
        "residual": True,   # not saved in checkpoint — must be forced
        "base_filters": 16,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mse_row(label: str, mse: dict) -> str:
    """Format one results row for the markdown table."""
    return (
        f"| {label} "
        f"| {mse['R']:.6f} "
        f"| {mse['G']:.6f} "
        f"| {mse['B']:.6f} "
        f"| **{mse['avg']:.6f}** |"
    )


def run_all(night_path: str, day_path: str) -> tuple[dict, list[dict]]:
    """
    Returns:
        baseline_mse  — channel-wise MSE dict for night vs day (no model)
        model_results — list of {name, mse} dicts in MODELS order
    """
    print("Loading eval images…")
    night_arr = load_image_rgb(night_path)
    day_arr   = load_image_rgb(day_path)

    # Baseline: night loaded exactly as model inputs, compared to day
    print("Computing baseline MSE (night.jpg vs day.jpg, no model)…")
    baseline_mse = channel_mse_eval(night_arr, day_arr)
    print(
        f"  Baseline → R={baseline_mse['R']:.6f}  G={baseline_mse['G']:.6f}"
        f"  B={baseline_mse['B']:.6f}  avg={baseline_mse['avg']:.6f}"
    )

    model_results = []
    for cfg in MODELS:
        ckpt = cfg["checkpoint"]
        if not Path(ckpt).exists():
            print(f"  [SKIP] {cfg['name']} — checkpoint not found: {ckpt}")
            model_results.append({"name": cfg["name"], "mse": None, "skipped": True})
            continue

        print(f"Running inference: {cfg['name']}…")
        enhanced = enhance(
            input_path=night_path,
            checkpoint_path=ckpt,
            base_filters=cfg["base_filters"],
            residual_override=cfg["residual"],
        )
        mse = channel_mse_eval(enhanced, day_arr)
        print(
            f"  {cfg['name']} → R={mse['R']:.6f}  G={mse['G']:.6f}"
            f"  B={mse['B']:.6f}  avg={mse['avg']:.6f}"
        )
        model_results.append({"name": cfg["name"], "mse": mse, "skipped": False})

    return baseline_mse, model_results


def write_evals_md(baseline_mse: dict, model_results: list[dict], out_path: str = "evals.md"):
    today = date.today().strftime("%Y-%m-%d")

    lines = [
        "# Evaluation Results",
        "",
        f"**Eval pair:** `night.jpg` (input) / `day.jpg` (ground truth)  ",
        f"**Date:** {today}  ",
        "**Metric:** channel-wise MSE in float64, pixel values in [0, 1]  ",
        "**Baseline:** `night.jpg` loaded as float32 RGB — same pipeline as model input — compared directly to `day.jpg` (no enhancement)  ",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Model | MSE_R | MSE_G | MSE_B | MSE_avg |",
        "|-------|------:|------:|------:|--------:|",
        mse_row("**Baseline (night vs day, no model)**", baseline_mse),
    ]

    for r in model_results:
        if r.get("skipped"):
            lines.append(f"| {r['name']} | — | — | — | *checkpoint not found* |")
        else:
            lines.append(mse_row(r["name"], r["mse"]))

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        "- **v3 / v4 checkpoints** do not store `residual=True` in saved args; the flag is forced at inference time.",
        "- **v4** additionally uses `GlobalContextEncoder` (per-channel mean/std/p10 injected at bottleneck); auto-detected from checkpoint.",
        "- Padding: images with H or W not divisible by 16 are reflect-padded (bottom/right) before the U-Net and cropped back before MSE — the padded region never contributes to the score.",
        "- All MSE values are in the [0, 1] pixel scale (not 0–255).",
        "",
    ]

    Path(out_path).write_text("\n".join(lines))
    print(f"\nResults written to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute baseline + per-model eval MSE")
    parser.add_argument("--night", default="night.jpg", help="Night input image (default: night.jpg)")
    parser.add_argument("--day",   default="day.jpg",   help="Day reference image (default: day.jpg)")
    parser.add_argument("--output", default="evals.md", help="Output markdown file (default: evals.md)")
    args = parser.parse_args()

    for p in (args.night, args.day):
        if not Path(p).exists():
            raise FileNotFoundError(f"Image not found: {p}")

    baseline_mse, model_results = run_all(args.night, args.day)
    write_evals_md(baseline_mse, model_results, args.output)


if __name__ == "__main__":
    main()
