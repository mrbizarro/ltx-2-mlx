#!/usr/bin/env python3
"""Face-crop contact sheets for a two-arm decoder A/B, on the house protocol.

The protocol (``CODEX_EXPERIMENTS_LTX25.md`` §Ground rules): a **448x448 window** on
the 1024x576 frame, LANCZOS to **512x512** -- a *downscale*, so neither arm is
sharpened by the resampler -- at frames **20 / 60 / 100**, side by side, **control on
the left**. Plus a 4x pixel-peep crop (NEAREST, so what you see is the real pixels).

Both arms are cropped with the SAME window at a given frame; the window is detected on
the control arm only. Inputs are the raw uint8 ``(F, H, W, 3)`` arrays the decoders
produced, not the mp4s, so no codec sits between the decoder and the owner's eye.

    python scripts/face_ab_sheet.py --control conv.npy --arm diffusion.npy \\
        --control-label "conv decoder (shipped)" --arm-label "diffusion decoder" \\
        --out-dir crops/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for path in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def detect_face(frame: np.ndarray, min_side: int = 120) -> tuple[int, int, int, int] | None:
    """Largest frontal face, or None.

    ``equalizeHist`` before the cascade and a ``min_side`` floor, both load-bearing on
    this footage: without them the cascade returns an 86px false positive in the
    rain-streaked background and the crop lands on a window instead of a face.
    """
    try:
        import cv2
    except ImportError:
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    grey = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    faces = cascade.detectMultiScale(grey, scaleFactor=1.05, minNeighbors=5, minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def face_centre(frames: np.ndarray, fid: int, search: int = 8) -> tuple[float, float] | None:
    """Face centre at ``fid``; if the cascade misses that exact frame, scan outward.

    A turned head or closed eyes defeats the frontal cascade on a single frame, and a
    missed frame must not silently become a centre crop -- faces are the metric.
    """
    for d in range(0, search + 1):
        for fj in ({fid} if d == 0 else {fid - d, fid + d}):
            if 0 <= fj < len(frames):
                box = detect_face(frames[fj])
                if box is not None:
                    x, y, w, h = box
                    return x + w / 2, y + h / 2
    return None


def window_for(frames: np.ndarray, fid: int, side: int, fallback_centre: tuple[float, float] | None) -> tuple[int, int, int]:
    """``(x0, y0, side)`` -- a ``side``x``side`` window centred on the face if there is one."""
    h, w = frames[fid].shape[:2]
    side = min(side, h, w)
    centre = face_centre(frames, fid)
    if centre is None:
        centre = fallback_centre if fallback_centre is not None else (w / 2, h / 2)
    cx, cy = centre
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return x0, y0, side


def crop(frame: np.ndarray, x0: int, y0: int, side: int, out: int, resample) -> Image.Image:
    im = Image.fromarray(frame[y0 : y0 + side, x0 : x0 + side])
    return im.resize((out, out), resample)


def labelled_pair(left: Image.Image, right: Image.Image, left_label: str, right_label: str, title: str) -> Image.Image:
    pad, head, foot = 10, 34, 30
    w = left.width + right.width + pad * 3
    h = head + left.height + foot + pad
    sheet = Image.new("RGB", (w, h), "#0d1017")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 9), title, font=_font(17), fill="#e8ecf4")
    sheet.paste(left, (pad, head))
    sheet.paste(right, (pad * 2 + left.width, head))
    d.text((pad, head + left.height + 7), left_label, font=_font(15), fill="#c9a3ff")
    d.text((pad * 2 + left.width, head + left.height + 7), right_label, font=_font(15), fill="#6c9bff")
    return sheet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, type=Path, help="(F,H,W,3) uint8 npy -- goes on the LEFT")
    ap.add_argument("--arm", required=True, type=Path)
    ap.add_argument("--control-label", default="control")
    ap.add_argument("--arm-label", default="arm")
    ap.add_argument("--frames", default="20,60,100")
    ap.add_argument("--window", type=int, default=448)
    ap.add_argument("--out-size", type=int, default=512)
    ap.add_argument("--peep-window", type=int, default=128, help="pixel-peep window, shown at 4x NEAREST")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    a = np.load(args.control)
    b = np.load(args.arm)
    if a.shape != b.shape:
        raise SystemExit(f"arms disagree on shape: {a.shape} vs {b.shape}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = [int(v) for v in args.frames.split(",")]
    ids = [min(i, len(a) - 1) for i in ids]
    boxes: dict[int, list[int]] = {}
    rows: list[Image.Image] = []
    last_centre: tuple[int, int] | None = None

    for fid in ids:
        x0, y0, side = window_for(a, fid, args.window, last_centre)
        last_centre = (x0 + side / 2, y0 + side / 2)
        boxes[fid] = [x0, y0, side]
        left = crop(a[fid], x0, y0, side, args.out_size, Image.LANCZOS)
        right = crop(b[fid], x0, y0, side, args.out_size, Image.LANCZOS)
        pair = labelled_pair(
            left, right, f"LEFT: {args.control_label}", f"RIGHT: {args.arm_label}",
            f"frame {fid} - {side}px window on {a.shape[2]}x{a.shape[1]}, LANCZOS to {args.out_size}px (downscale)",
        )
        pair.save(args.out_dir / f"sbs_f{fid:03d}.png")
        rows.append(pair)

        # pixel peep: same centre, small window, NEAREST at 4x
        pw = args.peep_window
        px = int(min(max(x0 + side // 2 - pw // 2, 0), a.shape[2] - pw))
        py = int(min(max(y0 + int(side * 0.55) - pw // 2, 0), a.shape[1] - pw))
        pl = Image.fromarray(a[fid][py : py + pw, px : px + pw]).resize((pw * 4, pw * 4), Image.NEAREST)
        pr = Image.fromarray(b[fid][py : py + pw, px : px + pw]).resize((pw * 4, pw * 4), Image.NEAREST)
        peep = labelled_pair(
            pl, pr, f"LEFT: {args.control_label}", f"RIGHT: {args.arm_label}",
            f"frame {fid} - {pw}px window at ({px},{py}), 4x NEAREST (real pixels, no resampling)",
        )
        peep.save(args.out_dir / f"peep4x_f{fid:03d}.png")

    width = max(r.width for r in rows)
    sheet = Image.new("RGB", (width, sum(r.height for r in rows)), "#0d1017")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(args.out_dir / "sheet_face_ab.png")

    meta = {
        "frames": ids,
        "windows": boxes,
        "window_px": args.window,
        "out_px": args.out_size,
        "resample": "LANCZOS (downscale)",
        "control": str(args.control),
        "arm": str(args.arm),
        "mean_abs_diff_over_window": {
            str(fid): round(
                float(
                    np.mean(
                        np.abs(
                            a[fid][boxes[fid][1] : boxes[fid][1] + boxes[fid][2],
                                   boxes[fid][0] : boxes[fid][0] + boxes[fid][2]].astype(np.int16)
                            - b[fid][boxes[fid][1] : boxes[fid][1] + boxes[fid][2],
                                     boxes[fid][0] : boxes[fid][0] + boxes[fid][2]].astype(np.int16)
                        )
                    )
                ),
                3,
            )
            for fid in ids
        },
    }
    (args.out_dir / "crops.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
