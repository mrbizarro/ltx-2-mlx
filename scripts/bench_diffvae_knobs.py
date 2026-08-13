#!/usr/bin/env python3
"""Decode ONE latent with the diffusion video VAE at ONE knob setting, and measure it.

The decoder report (`ltx25_diffusion_decoder.md` §5) showed the shipped default is not
the measured optimum -- 768x448x121 went 52.4 s (budget 100) -> 45.5 s (budget 8,
the default) -> 39.8 s (budget 2) -- and `LTX2_NA3D_MAX_WASTE` has never been swept
below its shipped 2.5 at all. This script is the instrument for that sweep, one
configuration per process so the peak-memory number belongs to that configuration
alone.

    python scripts/bench_diffvae_knobs.py --diffusion-pack DIR --latent latent.npy \\
        --json out.json [--profile] [--video out.mp4] [--pixels out.npy]

``--profile`` splits stage 5 between the masked SDPA (``na3d``) and everything else --
the channel-wise Linears / SwiGLU -- because if stage 5 is not attention-dominated then
``LTX2_NA3D_MAX_WASTE`` cannot help and sweeping it is theatre. The split costs one extra
``mx.eval`` per attention call, so its wall time is reported separately from a clean run's.

Knobs are read from the environment by the decoder itself (``LTX2_DIFFVAE_BUDGET_GB``,
``LTX2_DIFFVAE_TEMPORAL_OVERLAP``, ``LTX2_DIFFVAE_TOKEN_CHUNK``, ``LTX2_NA3D_MAX_WASTE``,
``LTX2_NA3D_MIN_TILE``); this script only records what they were.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for pkg in ("ltx-core-mlx", "ltx-pipelines-mlx"):
    sys.path.insert(0, str(REPO / "packages" / pkg / "src"))

from ltx_core_mlx.model.video_vae import diffusion_decoder as dd  # noqa: E402
from ltx_core_mlx.model.video_vae import na3d as na3d_mod  # noqa: E402
from ltx_core_mlx.model.video_vae.diffusion_decoder import (  # noqa: E402
    load_diffusion_decoder_from_pack,
)

FFMPEG = "/opt/homebrew/bin/ffmpeg"

KNOBS = (
    "LTX2_DIFFVAE_BUDGET_GB",
    "LTX2_DIFFVAE_TEMPORAL_OVERLAP",
    "LTX2_DIFFVAE_TOKEN_CHUNK",
    "LTX2_NA3D_MAX_WASTE",
    "LTX2_NA3D_MIN_TILE",
)


def to_uint8(pixels_bcfhw: mx.array) -> np.ndarray:
    """(B,3,F,H,W) in [-1,1] -> (F,H,W,3) uint8."""
    arr = np.asarray(pixels_bcfhw.astype(mx.float32))[0].transpose(1, 2, 3, 0)
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


def write_video(pixels: np.ndarray, path: Path, fps: float) -> None:
    f, h, w, _ = pixels.shape
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
           "-crf", "12", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=pixels.tobytes(), check=True, capture_output=True)


def stage5_tile_plan(t: int, h: int, w: int, kernel=(11, 11, 11)) -> dict:
    """What the current env's waste target does to the stage-5 query tiles."""
    tile = na3d_mod._pick_tiles(
        (t, h, w),
        kernel,
        na3d_mod.NA_SCORE_BUDGET,
        max_waste=float(os.environ.get("LTX2_NA3D_MAX_WASTE", na3d_mod.NA_MAX_WINDOW_WASTE)),
        min_tile_tokens=int(float(os.environ.get("LTX2_NA3D_MIN_TILE", na3d_mod.NA_MIN_TILE_TOKENS))),
    )
    keys = 1
    need = 1
    for d, k, tl in zip((t, h, w), kernel, tile):
        kk = min(k, d)
        keys *= min(d, tl + kk - 1)
        need *= kk
    n_tiles = 1
    for d, tl in zip((t, h, w), tile):
        n_tiles *= (d + tl - 1) // tl
    return {
        "grid": [t, h, w],
        "tile": list(tile),
        "keys_per_query": keys,
        "kernel_volume": need,
        "waste": round(keys / need, 3),
        "query_tiles": n_tiles,
    }


class Profiler:
    """Attribute stage-5 time: masked SDPA vs the channel-wise Linears."""

    def __init__(self) -> None:
        self.t_na3d = 0.0
        self.t_attn = 0.0
        self.t_mlp = 0.0
        self.t_ctx = 0.0
        self.n_na3d = 0
        self.pre_diffusion = 0.0
        self.diff_step = 0.0
        self.stage5_grid: list[int] | None = None

    def install(self) -> None:
        prof = self
        real_na3d = dd.na3d
        real_attn = dd.NeighborhoodAttention3D.__call__
        real_mlp = dd.SwiGLU.__call__
        real_block = dd.DiffusionNABlock.__call__
        real_pre = dd.NADiffusionDecoder.forward_pre_diffusion
        real_step = dd.NADiffusionDecoder.forward_diff_step
        # Only stage-5 modules are timed; the deterministic stages get one bulk number.
        in_stage5 = {"on": False}

        def na3d_timed(q, k, v, kernel, **kw):
            if not in_stage5["on"]:
                return real_na3d(q, k, v, kernel, **kw)
            if prof.stage5_grid is None:
                prof.stage5_grid = [int(q.shape[1]), int(q.shape[2]), int(q.shape[3])]
            t0 = time.perf_counter()
            out = real_na3d(q, k, v, kernel, **kw)
            mx.eval(out)
            prof.t_na3d += time.perf_counter() - t0
            prof.n_na3d += 1
            return out

        def attn_timed(self, x, pre=None):
            if not in_stage5["on"]:
                return real_attn(self, x, pre=pre)
            t0 = time.perf_counter()
            out = real_attn(self, x, pre=pre)
            mx.eval(out)
            prof.t_attn += time.perf_counter() - t0
            return out

        def mlp_timed(self, x, pre=None):
            if not in_stage5["on"]:
                return real_mlp(self, x, pre=pre)
            t0 = time.perf_counter()
            out = real_mlp(self, x, pre=pre)
            mx.eval(out)
            prof.t_mlp += time.perf_counter() - t0
            return out

        def block_timed(self, x, latent_context, modulation):
            t0 = time.perf_counter()
            out = real_block(self, x, latent_context, modulation)
            mx.eval(out)
            prof.t_ctx += time.perf_counter() - t0
            return out

        def pre_timed(self, *a, **kw):
            t0 = time.perf_counter()
            out = real_pre(self, *a, **kw)
            mx.eval(out)
            prof.pre_diffusion += time.perf_counter() - t0
            return out

        def step_timed(self, *a, **kw):
            in_stage5["on"] = True
            t0 = time.perf_counter()
            out = real_step(self, *a, **kw)
            mx.eval(out)
            prof.diff_step += time.perf_counter() - t0
            in_stage5["on"] = False
            return out

        dd.na3d = na3d_timed
        dd.NeighborhoodAttention3D.__call__ = attn_timed
        dd.SwiGLU.__call__ = mlp_timed
        dd.DiffusionNABlock.__call__ = block_timed
        dd.NADiffusionDecoder.forward_pre_diffusion = pre_timed
        dd.NADiffusionDecoder.forward_diff_step = step_timed

    def report(self) -> dict:
        block_total = self.t_ctx
        other = block_total - self.t_attn - self.t_mlp
        attn_other = self.t_attn - self.t_na3d
        return {
            "stages_1_4_seconds": round(self.pre_diffusion, 2),
            "stage5_seconds": round(self.diff_step, 2),
            "stage5_blocks_seconds": round(block_total, 2),
            "stage5_na3d_seconds": round(self.t_na3d, 2),
            "stage5_attn_seconds": round(self.t_attn, 2),
            "stage5_attn_qkv_rope_proj_seconds": round(attn_other, 2),
            "stage5_mlp_swiglu_seconds": round(self.t_mlp, 2),
            "stage5_context_proj_and_residual_seconds": round(other, 2),
            "na3d_calls": self.n_na3d,
            "stage5_grid": self.stage5_grid,
            "na3d_share_of_stage5": round(self.t_na3d / self.diff_step, 4) if self.diff_step else None,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diffusion-pack", type=Path, help="dir with vae_decoder_diffusion.safetensors")
    ap.add_argument("--conv-pack", type=Path, help="decode with the conv VideoDecoder from this pack instead")
    ap.add_argument("--latent", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--pixels", type=Path, help="save the raw uint8 (F,H,W,3) array")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--compare",
        type=Path,
        help="reference (F,H,W,3) uint8 npy: report the hash match and, when it does not "
        "match, how far apart the two pixel arrays actually are",
    )
    args = ap.parse_args(argv)

    if not args.diffusion_pack and not args.conv_pack:
        ap.error("pass --diffusion-pack or --conv-pack")

    latent = mx.array(np.load(args.latent))
    out: dict = {
        "label": args.label,
        "mlx_version": mx.__version__,
        "latent_shape": list(latent.shape),
        "arm": "conv" if args.conv_pack else "diffusion",
        "seed": args.seed,
        "env": {k: os.environ.get(k) for k in KNOBS},
    }

    prof = None
    if args.conv_pack:
        from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder
        from ltx_core_mlx.utils.weights import load_split_safetensors

        dec = VideoDecoder()
        dec.load_weights(
            list(load_split_safetensors(args.conv_pack / "vae_decoder.safetensors", prefix="vae_decoder.").items())
        )
        dec.eval()
        call = lambda: dec.decode(latent)  # noqa: E731
    else:
        if args.profile:
            prof = Profiler()
            prof.install()
        dec = load_diffusion_decoder_from_pack(args.diffusion_pack)
        b, c, f_lat, h_lat, w_lat = latent.shape
        tiles = dec.plan_temporal_tiles(f_lat, h_lat, w_lat)
        out["temporal_tiles"] = [list(t) for t in tiles]
        out["n_temporal_tiles"] = len(tiles)
        # stage-5 grid for the biggest tile: 8 frames per latent frame, /4 patch on 32x
        t_lat = max(t1 - t0 for t0, t1 in tiles)
        out["stage5_tile_plan"] = stage5_tile_plan(
            t_lat * 8, h_lat * 32 // 4, w_lat * 32 // 4
        )
        call = lambda: dec.decode(latent, seed=args.seed)  # noqa: E731

    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    pixels = call()
    mx.eval(pixels)
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    out["peak_gib"] = round(mx.get_peak_memory() / 1024**3, 3)

    rgb = to_uint8(pixels)
    out["pixels_shape"] = list(rgb.shape)
    out["pixels_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
    out["pixels_mean"] = round(float(rgb.mean()), 4)

    if prof is not None:
        out["profile"] = prof.report()
    if args.compare:
        ref = np.load(args.compare)
        same = hashlib.sha256(ref.tobytes()).hexdigest() == out["pixels_sha256"]
        cmp_out = {"reference": str(args.compare), "sha256_match": same}
        if not same and ref.shape == rgb.shape:
            diff = np.abs(ref.astype(np.int16) - rgb.astype(np.int16))
            mse = float(np.mean(diff.astype(np.float64) ** 2))
            cmp_out.update(
                mean_abs_diff_255=round(float(diff.mean()), 4),
                max_abs_diff_255=int(diff.max()),
                changed_pixel_fraction=round(float((diff > 0).mean()), 6),
                psnr_db=round(float(10 * np.log10(255**2 / mse)), 2) if mse else None,
            )
        out["compare"] = cmp_out
    if args.pixels:
        np.save(args.pixels, rgb)
    if args.video:
        write_video(rgb, args.video, args.fps)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
