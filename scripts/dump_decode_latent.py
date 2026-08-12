#!/usr/bin/env python3
"""Run any ``ltx-2-mlx`` CLI command and save the latent the video VAE is handed.

The A/B between the two video decoders needs ONE latent decoded twice. Rendering
twice would compare two different videos, not two decoders. So this wraps the normal
CLI, intercepts ``VideoDecoder.decode`` / ``decode_and_stream`` on the way in, writes
the latent to ``--latent-out``, and then lets the render finish exactly as it would
have. Nothing in the pipeline packages is modified.

    python scripts/dump_decode_latent.py --latent-out latent.npy -- \\
        generate --distilled --mode t2v --model PACK --prompt "..." ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for pkg in ("ltx-core-mlx", "ltx-pipelines-mlx"):
    sys.path.insert(0, str(REPO / "packages" / pkg / "src"))

import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402

from ltx_core_mlx.model.video_vae import video_vae  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-out", required=True, type=Path)
    args, rest = ap.parse_known_args(argv)
    if rest and rest[0] == "--":
        rest = rest[1:]

    saved: list[int] = []

    def save(latent: mx.array) -> None:
        if saved:
            return
        saved.append(1)
        arr = np.asarray(latent.astype(mx.float32))
        args.latent_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.latent_out, arr)
        print(f"[dump_decode_latent] saved {arr.shape} -> {args.latent_out}", flush=True)

    orig_decode = video_vae.VideoDecoder.decode
    orig_stream = video_vae.VideoDecoder.decode_and_stream

    def decode(self, latent, **kw):
        save(latent)
        return orig_decode(self, latent, **kw)

    def decode_and_stream(self, latent, *a, **kw):
        save(latent)
        return orig_stream(self, latent, *a, **kw)

    video_vae.VideoDecoder.decode = decode
    video_vae.VideoDecoder.decode_and_stream = decode_and_stream

    from ltx_pipelines_mlx.cli import main as cli_main

    # ``cli.main()`` takes no arguments — it parses ``sys.argv`` itself. Handing it a list
    # raised ``TypeError: main() takes 0 positional arguments``, i.e. this wrapper could
    # never have run against the current CLI. Rewrite argv instead.
    sys.argv = [sys.argv[0], *rest]
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
