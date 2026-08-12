#!/usr/bin/env python3
"""Build the labelled preview-evolution strip and score it against the delivered frame.

The whole feature rests on one claim: **the first forward's thumbnail already shows the
composition you are going to get.** This renders that claim as a picture — every published
preview, in order, with its sigma and stage, then the delivered frame on the right — and
attaches the number behind it: composition correlation (Pearson on a 40x24 luma downsample,
deliberately a composition measure rather than a texture one) against the delivered frame.

    python scripts/live_preview_strip.py --live-dir out/live --video out.mp4 \\
        --out strip.png --report strip.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for pkg in ("ltx-core-mlx", "ltx-pipelines-mlx"):
    sys.path.insert(0, str(REPO / "packages" / pkg / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


def composition_correlation(a: np.ndarray, b: np.ndarray, grid: tuple[int, int] = (24, 40)) -> float:
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


def read_frame(video: Path, index: int) -> np.ndarray:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    width, height = (int(v) for v in probe.split("x"))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{index})",
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-dir", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--panel-height", type=int, default=288)
    args = ap.parse_args()

    status = json.loads((args.live_dir / "status.json").read_text())
    sigmas = {}
    history = args.live_dir / "history.jsonl"
    if history.exists():
        for line in history.read_text().splitlines():
            row = json.loads(line)
            sigmas[row["forward"]] = row
    frame_index = status.get("approx_output_frame", 0)
    delivered = read_frame(args.video, frame_index).astype(np.float32) / 255.0

    previews = sorted(p for p in args.live_dir.glob("preview_*.png") if "latest" not in p.name)
    rows = []
    panels = []
    for path in previews:
        forward = int(path.stem.split("_")[1])
        image = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
        resized = np.asarray(
            Image.open(path).convert("RGB").resize(delivered.shape[1::-1], Image.LANCZOS)
        ).astype(np.float32) / 255.0
        meta = sigmas.get(forward, {})
        rows.append(
            {
                "forward": forward,
                "stage": meta.get("stage"),
                "sigma": meta.get("sigma"),
                "composition_correlation": round(composition_correlation(image, delivered), 4),
                "mean_abs_diff_255": round(float(np.abs(resized - delivered).mean() * 255.0), 2),
                "preview": str(path),
            }
        )
        label = f"fwd {forward}"
        if meta.get("stage"):
            label += f" {meta['stage']}"
        if meta.get("sigma") is not None:
            label += f"  s={meta['sigma']:.3f}"
        label += f"  corr {rows[-1]['composition_correlation']:.3f}"
        panels.append((Image.open(path).convert("RGB"), label))

    panels.append((Image.fromarray((delivered * 255).astype(np.uint8)), f"DELIVERED frame {frame_index}"))

    band = 22
    scaled = []
    for image, label in panels:
        ratio = args.panel_height / image.height
        scaled.append((image.resize((max(1, int(image.width * ratio)), args.panel_height), Image.LANCZOS), label))
    width = sum(image.width for image, _ in scaled)
    strip = Image.new("RGB", (width, args.panel_height + band), "black")
    draw = ImageDraw.Draw(strip)
    x = 0
    for image, label in scaled:
        strip.paste(image, (x, band))
        draw.text((x + 4, 5), label, fill="white")
        x += image.width
    args.out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(args.out)
    print(f"strip -> {args.out}  ({strip.width}x{strip.height}, {len(scaled)} panels)")

    for row in rows:
        print(f"  forward {row['forward']:2d} {str(row['stage']):8s} sigma {row['sigma']}  "
              f"corr {row['composition_correlation']:.4f}  mean|diff| {row['mean_abs_diff_255']}/255")

    if args.report:
        args.report.write_text(
            json.dumps({"delivered_frame": frame_index, "strip": str(args.out), "panels": rows}, indent=2) + "\n"
        )
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
