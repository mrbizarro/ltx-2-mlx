#!/usr/bin/env python3
"""Decode ONE latent with both LTX-2.5 video decoders and put the results side by side.

The two decoders share an encoder and a latent space -- the encoder halves of
``ltx-2.5-video-vae-bf16.safetensors`` and ``ltx-2.5-video-vae-conv-bf16.safetensors``
are byte-identical -- so this is a clean A/B: same latent in, two pictures out.

    # from a saved latent (npy / npz / safetensors, shape (B, C, F, H, W))
    python scripts/diffusion_decoder_ab.py --pack PACK --diffusion-pack DIFF \\
        --latent latent.npy --out ab_out

    # or round-trip real footage through the shared encoder first
    python scripts/diffusion_decoder_ab.py --pack PACK --diffusion-pack DIFF \\
        --from-video clip.mp4 --frames 25 --out ab_out

Reports wall time and MLX peak memory for each arm, writes both clips, writes matched
PNG frames, and -- because faces are the metric on this project -- writes a face-crop
contact sheet with the same crop taken from both arms.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for pkg in ("ltx-core-mlx", "ltx-pipelines-mlx"):
    sys.path.insert(0, str(REPO / "packages" / pkg / "src"))

from ltx_core_mlx.model.video_vae.diffusion_decoder import (  # noqa: E402
    load_diffusion_decoder_from_pack,
)
from ltx_core_mlx.model.video_vae.ops import remap_encoder_weight_keys  # noqa: E402
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder, VideoEncoder  # noqa: E402
from ltx_core_mlx.utils.weights import load_split_safetensors  # noqa: E402

FFMPEG = "/opt/homebrew/bin/ffmpeg"


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def read_video(path: Path, frames: int, width: int | None, height: int | None) -> np.ndarray:
    """Return ``(F, H, W, 3)`` uint8, optionally rescaled to a multiple of 32."""
    probe = subprocess.run(
        ["/opt/homebrew/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    w = width or (stream["width"] // 32) * 32
    h = height or (stream["height"] // 32) * 32
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}",
           "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    return np.frombuffer(raw, np.uint8)[: n * w * h * 3].reshape(n, h, w, 3)


def write_video(pixels: np.ndarray, path: Path, fps: float = 24.0) -> None:
    """``pixels``: ``(F, H, W, 3)`` uint8."""
    f, h, w, _ = pixels.shape
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
           "-crf", "12", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=pixels.tobytes(), check=True, capture_output=True)


def to_uint8(pixels_bcfhw: mx.array) -> np.ndarray:
    """(B,3,F,H,W) in [-1,1] -> (F,H,W,3) uint8."""
    arr = np.asarray(pixels_bcfhw.astype(mx.float32))[0].transpose(1, 2, 3, 0)
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def timed_decode(name: str, fn) -> tuple[np.ndarray, float, float]:
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = fn()
    mx.eval(out)
    wall = time.perf_counter() - t0
    peak = mx.get_peak_memory() / 1024**3
    print(f"  {name:<22} {wall:8.1f} s   peak {peak:6.2f} GiB", flush=True)
    return to_uint8(out), wall, peak


# ---------------------------------------------------------------------------
# face crops
# ---------------------------------------------------------------------------


def detect_face_box(frame: np.ndarray, pad: float = 0.55) -> tuple[int, int, int, int] | None:
    """Largest face in ``frame`` (H,W,3 uint8) as ``(x, y, w, h)``, padded out."""
    try:
        import cv2
    except ImportError:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + pad)
    H, W = frame.shape[:2]
    x0 = int(max(0, min(W - 1, cx - side / 2)))
    y0 = int(max(0, min(H - 1, cy - side / 2)))
    side = int(min(side, W - x0, H - y0))
    return x0, y0, side, side


def crop_sheet(arms: dict[str, np.ndarray], box, frame_ids, out: Path, zoom: int = 3) -> None:
    """One row per decoder, one column per frame, same crop everywhere."""
    from PIL import Image, ImageDraw

    x, y, w, h = box
    cell = (w * zoom, h * zoom)
    label_h = 26
    names = list(arms)
    sheet = Image.new("RGB", (cell[0] * len(frame_ids), (cell[1] + label_h) * len(names)), "black")
    draw = ImageDraw.Draw(sheet)
    for r, name in enumerate(names):
        for c, fid in enumerate(frame_ids):
            crop = Image.fromarray(arms[name][fid][y : y + h, x : x + w])
            crop = crop.resize(cell, Image.NEAREST)
            sheet.paste(crop, (c * cell[0], r * (cell[1] + label_h) + label_h))
        draw.text((6, r * (cell[1] + label_h) + 6), f"{name}  (crop {x},{y} {w}x{h}, {zoom}x nearest)", fill="white")
    sheet.save(out)


# ---------------------------------------------------------------------------


def sharpness(frame: np.ndarray, box=None) -> float:
    """Variance of the Laplacian -- a blunt but standard sharpness proxy."""
    import cv2

    if box:
        x, y, w, h = box
        frame = frame[y : y + h, x : x + w]
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, type=Path, help="pack dir with vae_decoder/vae_encoder.safetensors")
    ap.add_argument("--diffusion-pack", required=True, type=Path, help="dir with vae_decoder_diffusion.safetensors")
    ap.add_argument("--latent", type=Path, help="saved latent (B,C,F,H,W), .npy/.npz/.safetensors")
    ap.add_argument("--from-video", type=Path, help="encode this clip through the shared encoder instead")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", help="x,y,w,h face crop; omit to auto-detect")
    ap.add_argument("--sheet-frames", default="", help="comma-separated frame indices for the sheet")
    ap.add_argument("--only", choices=["conv", "diffusion"], help="run one arm only")
    ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu",
                    help="cpu is for correctness checks while another agent holds the GPU lock")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    if args.device == "cpu":
        mx.set_default_device(mx.cpu)

    args.out.mkdir(parents=True, exist_ok=True)
    source = None

    # ---- latent -----------------------------------------------------------
    if args.latent:
        if args.latent.suffix == ".npy":
            latent = mx.array(np.load(args.latent))
        elif args.latent.suffix == ".npz":
            blob = np.load(args.latent)
            latent = mx.array(blob[list(blob.files)[0]])
        else:
            blob = mx.load(str(args.latent))
            latent = next(iter(blob.values()))
        print(f"latent {tuple(latent.shape)} from {args.latent}")
    elif args.from_video:
        frames = read_video(args.from_video, args.frames, args.width, args.height)
        # 8k+1 temporal grid.
        n = max(1, ((len(frames) - 1) // 8) * 8 + 1)
        frames = frames[:n]
        source = frames
        print(f"encoding {frames.shape} through the shared VAE encoder")
        enc = VideoEncoder()
        weights = remap_encoder_weight_keys(
            load_split_safetensors(args.pack / "vae_encoder.safetensors", prefix="vae_encoder.")
        )
        enc.load_weights(list(weights.items()))
        enc.eval()
        pix = mx.array(frames.astype(np.float32) / 127.5 - 1.0)[None].transpose(0, 4, 1, 2, 3)
        latent = enc.encode(pix.astype(mx.bfloat16))
        mx.eval(latent)
        del enc
        mx.clear_cache()
        np.save(args.out / "latent.npy", np.asarray(latent.astype(mx.float32)))
        print(f"latent {tuple(latent.shape)} -> {args.out / 'latent.npy'}")
    else:
        ap.error("pass --latent or --from-video")

    arms: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}

    # ---- conv arm ---------------------------------------------------------
    if args.only in (None, "conv"):
        dec = VideoDecoder()
        dec.load_weights(
            list(load_split_safetensors(args.pack / "vae_decoder.safetensors", prefix="vae_decoder.").items())
        )
        dec.eval()
        pixels, wall, peak = timed_decode("conv decoder", lambda: dec.decode(latent))
        arms["conv decoder (default)"] = pixels
        stats["conv"] = {"seconds": wall, "peak_gib": peak, "shape": list(pixels.shape)}
        write_video(pixels, args.out / "conv.mp4", args.fps)
        del dec
        mx.clear_cache()

    # ---- diffusion arm ----------------------------------------------------
    if args.only in (None, "diffusion"):
        ddec = load_diffusion_decoder_from_pack(args.diffusion_pack)
        pixels, wall, peak = timed_decode("diffusion decoder", lambda: ddec.decode(latent, seed=args.seed))
        arms["diffusion decoder (opt-in)"] = pixels
        stats["diffusion"] = {"seconds": wall, "peak_gib": peak, "shape": list(pixels.shape)}
        write_video(pixels, args.out / "diffusion.mp4", args.fps)
        del ddec
        mx.clear_cache()

    if source is not None and len(arms) > 1:
        arms = {"source (ground truth)": source[: min(len(source), *(len(a) for a in arms.values()))], **arms}

    # ---- frames + sheet ---------------------------------------------------
    n_frames = min(len(a) for a in arms.values())
    ids = [int(i) for i in args.sheet_frames.split(",") if i.strip()] or [
        0, n_frames // 3, 2 * n_frames // 3, n_frames - 1
    ]
    ids = [min(i, n_frames - 1) for i in ids]

    from PIL import Image

    for name, arr in arms.items():
        tag = name.split()[0]
        for i in ids:
            Image.fromarray(arr[i]).save(args.out / f"frame{i:03d}_{tag}.png")

    box = None
    if args.crop:
        box = tuple(int(v) for v in args.crop.split(","))
    else:
        for i in ids:
            box = detect_face_box(next(iter(arms.values()))[i])
            if box:
                print(f"face detected on frame {i}: {box}")
                break
    if box:
        crop_sheet(arms, box, ids, args.out / "face_crops.png")
        for name, arr in arms.items():
            stats.setdefault("sharpness", {})[name] = round(
                float(np.mean([sharpness(arr[i], box) for i in ids])), 1
            )
    else:
        print("no face found; writing a centre-crop sheet instead")
        h, w = next(iter(arms.values())).shape[1:3]
        box = (w // 3, h // 5, w // 3, w // 3)
        crop_sheet(arms, box, ids, args.out / "centre_crops.png")

    stats["crop"] = list(box)
    (args.out / "ab_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
