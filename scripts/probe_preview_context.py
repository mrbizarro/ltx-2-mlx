#!/usr/bin/env python3
"""Falsify the live preview's two assumptions on a REAL generated LTX-2.5 latent.

Two questions, both answerable without touching the DiT:

**(c) Does a ``taeltx2_3`` checkpoint decode 2.5 latents legibly at all?**
Decode the latent the render actually handed the conv VAE, and compare frame by frame
against the delivered mp4. The measure is *composition correlation* — Pearson correlation of
both frames downsampled to 40x24 luma — deliberately a composition measure, not a texture
one, because that is the only thing a 22 MB preview decoder is being asked to get right.

**(d) How much causal context does one previewed frame need?**
Every ``MemBlock`` in the tiny decoder remembers the previous frame's input, so a frame
decoded alone is decoded as if it opened the clip. Decode the target latent frame with
``k`` warm-up frames for a sweep of ``k`` and measure mean |diff| against the same frame
taken from the full-sequence decode (the ground truth for "what this decoder would say").

    python scripts/probe_preview_context.py \\
        --latent latent.npy --video draft.mp4 --tae taeltx2_3.safetensors \\
        --out-dir notes/ltx25_live_preview
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for pkg in ("ltx-core-mlx", "ltx-pipelines-mlx"):
    sys.path.insert(0, str(REPO / "packages" / pkg / "src"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from ltx_pipelines_mlx.live_preview import local_output_index  # noqa: E402
from ltx_pipelines_mlx.tiny_video_vae import load_tiny_ltx_video_decoder  # noqa: E402


def composition_correlation(a: np.ndarray, b: np.ndarray, grid: tuple[int, int] = (24, 40)) -> float:
    """Pearson correlation of two HxWx3 frames downsampled to a ``grid`` luma thumbnail."""

    def thumb(x: np.ndarray) -> np.ndarray:
        luma = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
        h, w = luma.shape
        gh, gw = grid
        ys = np.linspace(0, h, gh + 1).astype(int)
        xs = np.linspace(0, w, gw + 1).astype(int)
        return np.array(
            [[luma[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].mean() for j in range(gw)] for i in range(gh)]
        ).ravel()

    u, v = thumb(a), thumb(b)
    u = u - u.mean()
    v = v - v.mean()
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / denom) if denom else 0.0


def read_video_frames(path: Path) -> np.ndarray:
    """Decode an mp4 to ``(T, H, W, 3)`` uint8 via ffmpeg."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    width, height = (int(v) for v in probe.split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True, type=Path, help="(1,128,F,H,W) .npy from dump_decode_latent.py")
    ap.add_argument("--video", required=True, type=Path, help="the mp4 the conv VAE produced from that latent")
    ap.add_argument("--tae", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 8])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    latent_np = np.load(args.latent)
    latent = mx.array(latent_np.astype(np.float32))
    print(f"latent {latent.shape}  mean={float(latent.mean()):+.4f}  std={float(latent.std()):.4f}")
    _, channels, frames, lat_h, lat_w = latent.shape

    tae = load_tiny_ltx_video_decoder(args.tae)
    print(f"tae latent_channels={tae.latent_channels} patch={tae.patch_size} t_up={tae.temporal_upscale}")
    if channels != tae.latent_channels:
        raise SystemExit(f"latent width {channels} != decoder's {tae.latent_channels}")

    started = time.perf_counter()
    full = tae.decode(latent)
    mx.eval(full)
    full_seconds = time.perf_counter() - started
    full_np = np.array(full)[0].transpose(1, 2, 3, 0)  # (T, H, W, 3) in [0, 1]
    print(f"full-sequence TAE decode: {full_np.shape} in {full_seconds:.3f}s")

    delivered = read_video_frames(args.video).astype(np.float32) / 255.0
    print(f"delivered mp4: {delivered.shape}")

    report: dict = {
        "latent": str(args.latent),
        "latent_shape": list(latent_np.shape),
        "video": str(args.video),
        "tae": str(args.tae),
        "mlx_version": mx.__version__,
        "full_decode_seconds": round(full_seconds, 4),
    }

    # --- (c) does the tiny decoder see the same picture as the real one? ---
    checked = min(len(full_np), len(delivered))
    sample = sorted({0, checked // 4, checked // 2, (3 * checked) // 4, checked - 1})
    rows = []
    for i in sample:
        rows.append(
            {
                "frame": i,
                "composition_correlation": round(composition_correlation(full_np[i], delivered[i]), 4),
                "mean_abs_diff_255": round(float(np.abs(full_np[i] - delivered[i]).mean() * 255.0), 2),
            }
        )
        print(f"  frame {i:4d}  corr {rows[-1]['composition_correlation']:.4f}  "
              f"mean|diff| {rows[-1]['mean_abs_diff_255']:.2f}/255")
    report["tae_vs_delivered"] = rows

    # --- (d) how much causal warm-up does one frame need? ---
    target = frames // 2
    truth = full_np[local_output_index(target)] if target * 8 <= len(full_np) else full_np[8 * target - 7]
    truth = full_np[min(len(full_np) - 1, max(0, 8 * target - 7))]
    context_rows = []
    for k in args.contexts:
        if k > target:
            continue
        first = target - k
        piece = latent[:, :, first : target + 1]
        started = time.perf_counter()
        decoded = tae.decode(piece, local_output_index(k) + 1)
        mx.eval(decoded)
        seconds = time.perf_counter() - started
        frame = np.array(decoded)[0, :, local_output_index(k)].transpose(1, 2, 0)
        context_rows.append(
            {
                "context": k,
                "latent_tokens": k + 1,
                "seconds": round(seconds, 4),
                "mean_abs_diff_255": round(float(np.abs(frame - truth).mean() * 255.0), 3),
                "composition_correlation": round(composition_correlation(frame, truth), 5),
            }
        )
        print(f"  context {k}: {seconds:.3f}s  mean|diff| {context_rows[-1]['mean_abs_diff_255']:.3f}/255  "
              f"corr {context_rows[-1]['composition_correlation']:.5f}")
    report["context_sweep"] = context_rows
    report["context_truth_frame"] = int(min(len(full_np) - 1, max(0, 8 * target - 7)))
    report["previewed_latent_frame"] = target

    # A strip: naive (context 0) | shipped default | full-clip TAE | delivered
    from PIL import Image

    def to_img(arr: np.ndarray) -> Image.Image:
        return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))

    panels = []
    labels = []
    for k in (0, 2):
        if k > target:
            continue
        piece = latent[:, :, target - k : target + 1]
        decoded = tae.decode(piece, local_output_index(k) + 1)
        mx.eval(decoded)
        panels.append(to_img(np.array(decoded)[0, :, local_output_index(k)].transpose(1, 2, 0)))
        labels.append(f"context {k}")
    panels.append(to_img(truth))
    labels.append("full-clip TAE")
    panels.append(to_img(delivered[report["context_truth_frame"]]))
    labels.append("delivered (conv VAE)")

    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    strip = Image.new("RGB", (width, height), "black")
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width
    strip_path = args.out_dir / "PREVIEW_CONTEXT_PROBE.png"
    strip.save(strip_path)
    report["strip"] = str(strip_path)
    report["strip_panels"] = labels
    print(f"strip -> {strip_path}  panels: {labels}")

    out = args.out_dir / "preview_context_probe.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
