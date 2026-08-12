#!/usr/bin/env python3
"""Per-block quantization-error profile: q4 against q8, on the packs already on disk.

Written for ``CODEX_EXPERIMENTS_LTX25.md`` Experiment 4, **stage A** — the cheap
falsification that costs no GPU and no download. The hypothesis under test is
rockerBOO's (HF ``rockerBOO/ltx-2.5-nvfp4-convrot``): that q4's patch-like
artifacts come disproportionately from a few precision-sensitive transformer
blocks, and specifically that **blocks 0, 1, 46 and 47** are the outliers.

Method, and its one stated assumption:

    For every quantized Linear in every transformer block, dequantize the q4
    pack's weight and the q8 pack's weight and measure the difference, using
    **q8 as the bf16 proxy**. The bf16 source is not on disk (it was deleted
    after the mirror shipped), so this is a proxy and not the real thing; it is
    a defensible one because q8's own worst mean relative error against bf16 is
    0.0157 (``phosphene_quant_manifest.json`` validation probes), roughly 60x
    smaller than the q4 effect being measured.

Nothing here touches the GPU: the default device is forced to CPU, so this can
run while another agent holds both GPU locks. Nothing here loads a model — the
safetensors files are memory-mapped and read one tensor at a time, so peak RSS
stays at a few hundred MB against 32 GB of weights.

Usage::

    python scripts/block_quant_error_profile.py \\
        --q4 .../ltx-2.5-mlx-q4/transformer-distilled.safetensors \\
        --q8 .../ltx-2.5-mlx-q8/transformer-distilled.safetensors \\
        --json out/profile.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)

_BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")


class SafeTensorsView:
    """Read individual tensors out of a safetensors file without loading it.

    MLX's own loader materialises the whole dict, and ``safetensors``' numpy
    framework refuses bf16 outright ("data type 'bfloat16' not understood").
    Both are avoidable: the file is a JSON header followed by a flat byte
    region, so a memmap plus the header's offsets gives one tensor at a time,
    and bf16 is read as uint16 and re-viewed.
    """

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as fh:
            header_len = struct.unpack("<Q", fh.read(8))[0]
            self.header = json.loads(fh.read(header_len))
        self._base = 8 + header_len
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")

    def __contains__(self, key: str) -> bool:
        return key in self.header

    def keys(self) -> list[str]:
        return [k for k in self.header if k != "__metadata__"]

    def dtype_of(self, key: str) -> str:
        return self.header[key]["dtype"]

    def get(self, key: str) -> mx.array:
        entry = self.header[key]
        start, end = entry["data_offsets"]
        raw = self._mm[self._base + start : self._base + end]
        dtype = entry["dtype"]
        if dtype == "U32":
            return mx.array(np.frombuffer(raw, dtype=np.uint32).reshape(entry["shape"]))
        if dtype == "BF16":
            return mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(entry["shape"])).view(mx.bfloat16)
        if dtype == "F32":
            return mx.array(np.frombuffer(raw, dtype=np.float32).reshape(entry["shape"]))
        if dtype == "F16":
            return mx.array(np.frombuffer(raw, dtype=np.uint16).reshape(entry["shape"])).view(mx.float16)
        raise ValueError(f"{key}: unhandled dtype {dtype}")


def _bits_and_group(view: SafeTensorsView, prefix: str) -> tuple[int, int]:
    """(bits, group_size) for one quantized Linear, derived from shapes alone.

    ``weight`` is uint32-packed: ``packed_words = in_features * bits / 32``.
    ``scales`` is ``(out_features, in_features / group_size)``. Trying the
    plausible bit widths and keeping the one that yields a legal group_size
    reads the pack's own geometry instead of trusting ``quantize_config.json``
    — the same rule the loader uses (``utils/weights.py::derive_quant_params``).
    """
    packed = view.header[f"{prefix}.weight"]["shape"][-1]
    groups = view.header[f"{prefix}.scales"]["shape"][-1]
    for bits in (2, 3, 4, 6, 8):
        in_features = packed * 32 // bits
        if in_features % groups:
            continue
        group_size = in_features // groups
        if group_size in (32, 64, 128):
            return bits, group_size
    raise ValueError(f"{prefix}: cannot derive (bits, group_size) from packed={packed} groups={groups}")


def profile(q4_path: Path, q8_path: Path, limit_blocks: int | None = None) -> dict:
    t0 = time.perf_counter()
    v4 = SafeTensorsView(q4_path)
    v8 = SafeTensorsView(q8_path)

    q4_keys = set(v4.keys())
    q8_keys = set(v8.keys())
    if q4_keys != q8_keys:
        only4 = sorted(q4_keys - q8_keys)[:10]
        only8 = sorted(q8_keys - q4_keys)[:10]
        raise SystemExit(f"key sets differ. q4-only {only4} ... q8-only {only8}")

    # Every module inside a transformer block that carries `.scales` in BOTH
    # packs is a quantized Linear. A module that carries a `.weight` but no
    # `.scales` was left in float — which is exactly what rockerBOO's recipe
    # does, so we count those separately rather than skipping them silently.
    quantized: dict[int, list[str]] = defaultdict(list)
    float_modules: dict[int, list[str]] = defaultdict(list)
    for key in sorted(q4_keys):
        m = _BLOCK_RE.search(key)
        if not m or not key.endswith(".weight"):
            continue
        block = int(m.group(1))
        prefix = key[: -len(".weight")]
        if f"{prefix}.scales" in q4_keys and f"{prefix}.scales" in q8_keys:
            quantized[block].append(prefix)
        elif v4.header[key]["shape"] and len(v4.header[key]["shape"]) == 2:
            float_modules[block].append(prefix)

    blocks = sorted(quantized)
    if limit_blocks is not None:
        blocks = blocks[:limit_blocks]

    modules: list[dict] = []
    per_block: dict[int, dict] = {}

    for block in blocks:
        b_sse = b_ss8 = b_l1 = b_absw = 0.0
        b_n = 0
        b_max = 0.0
        started = time.perf_counter()
        for prefix in sorted(quantized[block]):
            bits4, gs4 = _bits_and_group(v4, prefix)
            bits8, gs8 = _bits_and_group(v8, prefix)
            w4 = mx.dequantize(v4.get(f"{prefix}.weight"), v4.get(f"{prefix}.scales"), v4.get(f"{prefix}.biases"), group_size=gs4, bits=bits4).astype(mx.float32)
            w8 = mx.dequantize(v8.get(f"{prefix}.weight"), v8.get(f"{prefix}.scales"), v8.get(f"{prefix}.biases"), group_size=gs8, bits=bits8).astype(mx.float32)
            d = w4 - w8
            sse = float(mx.sum(d * d).item())
            ss8 = float(mx.sum(w8 * w8).item())
            l1 = float(mx.sum(mx.abs(d)).item())
            absw = float(mx.sum(mx.abs(w8)).item())
            mx_abs = float(mx.max(mx.abs(d)).item())
            # Outlier structure of the *reference* weight: quantization error in
            # a group scales with that group's max, so a heavy-tailed block is
            # mechanically worse at 4 bits. Reported so the ranking can be
            # explained rather than merely observed.
            rms = math.sqrt(ss8 / w8.size)
            peak = float(mx.max(mx.abs(w8)).item())
            modules.append(
                {
                    "block": block,
                    "module": prefix.split("transformer_blocks.")[-1].split(".", 1)[-1],
                    "prefix": prefix,
                    "bits_q4": bits4,
                    "bits_q8": bits8,
                    "group_size": gs4,
                    "numel": int(w8.size),
                    "rel_fro": math.sqrt(sse / ss8) if ss8 else float("nan"),
                    "rel_l1": (l1 / absw) if absw else float("nan"),
                    "max_abs_err": mx_abs,
                    "peak_over_rms": (peak / rms) if rms else float("nan"),
                }
            )
            b_sse += sse
            b_ss8 += ss8
            b_l1 += l1
            b_absw += absw
            b_n += int(w8.size)
            b_max = max(b_max, mx_abs)
            del w4, w8, d
        mx.clear_cache()
        per_block[block] = {
            "block": block,
            "modules": len(quantized[block]),
            "float_modules": len(float_modules.get(block, [])),
            "params": b_n,
            "rel_fro": math.sqrt(b_sse / b_ss8) if b_ss8 else float("nan"),
            "rel_l1": (b_l1 / b_absw) if b_absw else float("nan"),
            "max_abs_err": b_max,
            "seconds": round(time.perf_counter() - started, 2),
        }
        print(
            f"[block {block:2d}] rel_fro={per_block[block]['rel_fro']:.6f} "
            f"rel_l1={per_block[block]['rel_l1']:.6f} max={b_max:.6f} "
            f"({per_block[block]['seconds']}s)",
            flush=True,
        )

    ranked = sorted(per_block.values(), key=lambda r: -r["rel_fro"])
    edge = [0, 1, 46, 47]
    present_edge = [b for b in edge if b in per_block]
    rank_of = {r["block"]: i + 1 for i, r in enumerate(ranked)}

    # Per module family, averaged over blocks — tells us whether the error is a
    # property of the block or of the projection type.
    family: dict[str, list[float]] = defaultdict(list)
    for row in modules:
        family[row["module"]].append(row["rel_fro"])
    family_summary = sorted(
        ({"module": k, "mean_rel_fro": sum(v) / len(v), "n": len(v)} for k, v in family.items()),
        key=lambda r: -r["mean_rel_fro"],
    )

    return {
        "mlx_version": mx.__version__,
        "q4": str(q4_path),
        "q8": str(q8_path),
        "assumption": (
            "q8 stands in for bf16. q8's own worst mean rel err against bf16 is 0.0157 "
            "(pack manifest validation probes), ~60x below the q4-vs-q8 effect measured here."
        ),
        "blocks_profiled": len(blocks),
        "modules_compared": len(modules),
        "float_modules_in_blocks": {str(k): v for k, v in float_modules.items() if v},
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "per_block": [per_block[b] for b in blocks],
        "ranked_blocks": [r["block"] for r in ranked],
        "edge_block_ranks": {str(b): rank_of[b] for b in present_edge},
        "module_families": family_summary,
        "modules": modules,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q4", required=True, type=Path)
    ap.add_argument("--q8", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--limit-blocks", type=int, default=None, help="profile only the first N blocks (smoke test)")
    args = ap.parse_args()

    payload = profile(args.q4, args.q8, args.limit_blocks)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    ranked = payload["ranked_blocks"]
    print("\nblocks ranked by relative Frobenius error (worst first):")
    print("  " + " ".join(str(b) for b in ranked))
    print("\nrockerBOO's edge blocks and where they actually rank:")
    for b, r in payload["edge_block_ranks"].items():
        print(f"  block {b:>2}: rank {r} of {payload['blocks_profiled']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
