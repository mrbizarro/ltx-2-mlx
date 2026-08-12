#!/usr/bin/env python3
"""Measure what the unfused LoRA branch costs, per layer, against the fused base.

The claim in ``notes/ltx25_lora_investigation.md`` §7.1 is ~1.5 % extra FLOPs at
rank 32. That is closed-form:

    base   2·N·in·out
    lora   2·N·r·(in + out)
    ratio  r·(in + out) / (in·out)  =  32·8192 / 4096²  =  1.5625 %

This script checks the wall-clock consequence, which is the number that
actually matters — a 1.5 % FLOP increase can cost more than 1.5 % of the time
if it adds kernel launches to a memory-bound layer, and less if the GPU was
waiting on the quantized matmul anyway.

Layer-level, so it needs no GPU lock and no weights: it times the exact modules
the DiT runs (``nn.QuantizedLinear`` at the real widths and token counts) with
and without the adapter.

    python3 scripts/bench_runtime_lora.py
    python3 scripts/bench_runtime_lora.py --rank 64 --iters 100
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.loader.runtime_loras import LoRAQuantizedLinear

#: (label, tokens) — the sequence lengths a real render puts through attention.
#: 768×448×49 draft → 7 latent frames × 14 × 24 = 2352 video tokens.
#: 1024×576×121   → 16 latent frames × 18 × 32 = 9216 video tokens.
TOKEN_CASES = (("draft 768x448x49", 2352), ("panel 1024x576x121", 9216))

#: (label, in_features, out_features) — LTX-2's video and audio projections.
LAYER_CASES = (("video attn 4096", 4096, 4096), ("audio attn 2048", 2048, 2048))


def _chain(fn, x: mx.array, depth: int) -> mx.array:
    """``depth`` dependent applications of ``fn``, evaluated as one graph.

    A per-call ``mx.eval`` would charge every kernel launch and one host sync to
    each call, which over-weights the adapter (it dispatches three kernels where
    the base dispatches one) and understates how well MLX pipelines them. The
    DiT runs 48 blocks back to back inside a single lazy graph, so chaining is
    the honest shape for this measurement.
    """
    for _ in range(depth):
        x = fn(x)
    return x


def _time(fn, x: mx.array, iters: int, warmup: int, depth: int) -> float:
    """Best seconds per layer application over ``iters`` runs.

    Minimum, not mean: this box shares its GPU with renders, and contention can
    only ever make a run slower. The fastest observed run is the closest
    estimate of the cost with the machine to itself, and it is the estimator
    that does not drift with whoever else is working.
    """
    for _ in range(warmup):
        mx.eval(_chain(fn, x, depth))
    mx.synchronize()
    best = float("inf")
    for _ in range(iters):
        started = time.perf_counter()
        mx.eval(_chain(fn, x, depth))
        mx.synchronize()
        best = min(best, time.perf_counter() - started)
    return best / depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (default: 32, bizarrotrn_v2)")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bit width (default: 4)")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--depth", type=int, default=48, help="Dependent applications per timed graph (default: 48, the DiT's block count)"
    )
    args = parser.parse_args()

    mx.random.seed(0)
    print(
        f"rank {args.rank} · int{args.bits} g{args.group_size} · depth {args.depth} · "
        f"{args.iters} iters (+{args.warmup} warmup)\n"
    )
    header = f"{'layer':<18} {'tokens':<20} {'base ms':>9} {'unfused ms':>11} {'overhead':>9} {'FLOPs':>8}"
    print(header)
    print("-" * len(header))

    for layer_label, in_features, out_features in LAYER_CASES:
        linear = nn.Linear(in_features, out_features, bias=True)
        quantized = nn.QuantizedLinear.from_linear(linear, group_size=args.group_size, bits=args.bits)
        a = mx.random.normal((args.rank, in_features)).astype(mx.bfloat16)
        b = mx.random.normal((out_features, args.rank)).astype(mx.bfloat16)
        adapter = LoRAQuantizedLinear(quantized, a, b, 1.0)
        flops_ratio = args.rank * (in_features + out_features) / (in_features * out_features)

        for token_label, tokens in TOKEN_CASES:
            x = mx.random.normal((1, tokens, in_features)).astype(mx.bfloat16)
            base_s = _time(quantized, x, args.iters, args.warmup, args.depth)
            lora_s = _time(adapter, x, args.iters, args.warmup, args.depth)
            print(
                f"{layer_label:<18} {token_label:<20} {base_s * 1e3:>9.3f} {lora_s * 1e3:>11.3f} "
                f"{(lora_s / base_s - 1) * 100:>8.2f}% {flops_ratio * 100:>7.2f}%"
            )

    print(
        "\nPer-step cost = per-layer overhead x (the LoRA's share of the block's matmuls).\n"
        "bizarrotrn_v2 targets 24 of ~30 linears per block, so the block-level figure\n"
        "is close to the per-layer one; the DiT's non-linear work (norms, RoPE,\n"
        "softmax, VAE, text encoder) dilutes it further at the render level."
    )


if __name__ == "__main__":
    main()
