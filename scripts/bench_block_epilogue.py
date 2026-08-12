#!/usr/bin/env python3
"""Bench the LTX-2.5 DiT block's full-width epilogues, naive vs fused.

Experiment 6 of ``CODEX_EXPERIMENTS_LTX25.md``. The prior comes from
``notes/ltx25_unfused_lora.md`` §3: at these widths a layer is bound by the
output write, not by arithmetic — a separate ``y + delta`` writes an ``N x out``
tensor, reads it back and writes the sum, which cost **+17 %** wall clock for a
1.6 % FLOP increase until the add was folded into ``mx.addmm`` (**+2.7 %**).

This script asks whether the same lever exists at the DiT's *own* sites. Every
site that materialises a full-width ``N x out`` temporary and immediately
consumes it is benched in both forms, isolated, min-of-N, at the real geometry:

    A  attention out-projection + per-head gate + residual   h + to_out(x) * g
    B  feed-forward + gate + residual                        h + ff(x) * g
    C  AdaLN modulate                                        rms(h) * (1+s) + t
    D  the whole block, eager vs mx.compile                  the fusion CEILING

D is the honest upper bound: ``mx.compile`` fuses every elementwise chain in
the block automatically, so no hand-written epilogue can beat it. If D is under
the gate, the experiment is over regardless of what any single site shows.

**Instrument check** (arm ``I``): the *known* effect from the LoRA note,
reproduced here — a dense bf16 matmul with a separate add vs ``mx.addmm``. If
the bench cannot see that, a null result at A/B/C means nothing.

    python3 scripts/bench_block_epilogue.py
    python3 scripts/bench_block_epilogue.py --tokens 9216 --iters 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ltx_core_mlx.model.transformer.model import LTXModelConfig
from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions

CHECKPOINT = "/Users/salo/pinokio/api/phosphene-dev.git/mlx_models/ltx-2.5-mlx-q8/transformer-dev.safetensors"


def _chain(fn, x, depth: int):
    """``depth`` dependent applications, evaluated as one lazy graph.

    Same reasoning as ``bench_runtime_lora.py``: a per-call ``mx.eval`` charges
    a host sync to every call and over-weights whichever arm dispatches more
    kernels. The DiT runs 48 blocks back to back inside one graph.
    """
    for _ in range(depth):
        x = fn(x)
    return x


def _time(fn, x, iters: int, warmup: int, depth: int) -> float:
    """Best seconds per application. Minimum, not mean — contention only ever adds."""
    for _ in range(warmup):
        mx.eval(_chain(fn, x, depth))
    mx.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(_chain(fn, x, depth))
        mx.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best / depth


def _rel(a: mx.array, b: mx.array) -> float:
    d = (a.astype(mx.float32) - b.astype(mx.float32))
    n = mx.linalg.norm(d) / mx.maximum(mx.linalg.norm(b.astype(mx.float32)), 1e-30)
    return float(n)


def _bitexact(a: mx.array, b: mx.array) -> bool:
    return bool(mx.all(a == b).item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, nargs="*", default=[2304, 9216],
                    help="Video token counts (default: stage 1 half-res 2304 and stage 2 full-res 9216)")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--bits", type=int, default=8, help="Quantization of the DiT linears (q8 pack)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--gate", type=float, default=3.0, help="Adoption gate, %% on the isolated bench")
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-block", action="store_true", help="Skip arm D (the whole-block ceiling)")
    args = ap.parse_args()

    cfg = LTXModelConfig.from_checkpoint_file(CHECKPOINT) or LTXModelConfig()
    dim, adim = cfg.video_dim, cfg.audio_dim
    eps = cfg.norm_eps
    mx.random.seed(0)
    print(f"mlx {mx.__version__} · int{args.bits} g{args.group_size} · dim {dim} · "
          f"{args.iters} iters (+{args.warmup} warmup)\n")

    rows: list[dict] = []

    def report(site: str, tokens: int, naive_s: float, fused_s: float, label: str, exact: str) -> None:
        delta = (1.0 - fused_s / naive_s) * 100.0
        rows.append({"site": site, "tokens": tokens, "variant": label,
                     "naive_ms": naive_s * 1e3, "fused_ms": fused_s * 1e3,
                     "saved_pct": delta, "exactness": exact})
        print(f"{site:<34} {tokens:>6} {label:<22} {naive_s * 1e3:>9.3f} {fused_s * 1e3:>9.3f} "
              f"{delta:>+8.2f}%  {exact}")

    header = f"{'site':<34} {'tokens':>6} {'fused form':<22} {'naive ms':>9} {'fused ms':>9} {'saved':>9}  exactness"
    print(header)
    print("-" * len(header))

    # ---------------- I — instrument check: the LoRA note's known effect ----
    # Dense bf16 matmul, separate add vs mx.addmm. This effect is documented at
    # ~14 percentage points; if it does not show up here the bench is blind.
    for tokens in args.tokens:
        w = mx.random.normal((dim, dim)).astype(mx.bfloat16)
        c = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
        x = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
        mx.eval(w, c, x)
        w_t = w.T

        def naive(v, w_t=w_t, c=c):
            return c + (v @ w_t)

        c2 = c.reshape(-1, dim)

        def fused(v, w_t=w_t, c2=c2):
            return mx.addmm(c2, v.reshape(-1, dim), w_t).reshape(1, -1, dim)

        exact = "bit-exact" if _bitexact(naive(x), fused(x)) else f"rel {_rel(fused(x), naive(x)):.2e}"
        n_s = _time(naive, x, args.iters, args.warmup, 8)
        f_s = _time(fused, x, args.iters, args.warmup, 8)
        report("I instrument check (dense add)", tokens, n_s, f_s, "mx.addmm", exact)

    # ---------------- A — attention out-projection + gate + residual --------
    # h + to_out(x) * g.  to_out is a QuantizedLinear in every shipped pack, and
    # MLX has no quantized addmm (mx.quantized_matmul takes no C/beta), so the
    # LoRA note's lever is structurally unavailable here. What remains is
    # elementwise fusion of `* g` and `+ h`, which is what mx.compile does.
    for tokens in args.tokens:
        lin = nn.Linear(dim, dim, bias=True)
        q = nn.QuantizedLinear.from_linear(lin, group_size=args.group_size, bits=args.bits)
        g = mx.random.normal((1, 1, dim)).astype(mx.bfloat16)
        h = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
        x = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
        mx.eval(q.parameters(), g, h, x)

        def naive(v, q=q, g=g, h=h):
            return h + q(v) * g

        compiled = mx.compile(naive)
        exact = "bit-exact" if _bitexact(naive(x), compiled(x)) else f"rel {_rel(compiled(x), naive(x)):.2e}"
        n_s = _time(naive, x, args.iters, args.warmup, 12)
        f_s = _time(compiled, x, args.iters, args.warmup, 12)
        report("A attn out + gate + residual", tokens, n_s, f_s, "mx.compile", exact)

    # ---------------- B — feed-forward + gate + residual --------------------
    for tokens in args.tokens:
        inner = int(dim * cfg.ff_mult)
        pin = nn.QuantizedLinear.from_linear(
            nn.Linear(dim, inner, bias=cfg.ff_bias), group_size=args.group_size, bits=args.bits)
        pout = nn.QuantizedLinear.from_linear(
            nn.Linear(inner, dim, bias=cfg.ff_bias), group_size=args.group_size, bits=args.bits)
        g = mx.random.normal((1, 1, dim)).astype(mx.bfloat16)
        h = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
        x = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
        mx.eval(pin.parameters(), pout.parameters(), g, h, x)

        def naive(v, pin=pin, pout=pout, g=g, h=h):
            return h + pout(nn.gelu_approx(pin(v))) * g

        compiled = mx.compile(naive)
        exact = "bit-exact" if _bitexact(naive(x), compiled(x)) else f"rel {_rel(compiled(x), naive(x)):.2e}"
        n_s = _time(naive, x, args.iters, args.warmup, 4)
        f_s = _time(compiled, x, args.iters, args.warmup, 4)
        report("B ff + gate + residual", tokens, n_s, f_s, "mx.compile", exact)

    # ---------------- C — AdaLN modulate ------------------------------------
    # rms(h) * (1 + scale) + shift.  Three passes over the widest tensor.
    # Two candidate folds: mx.fast.rms_norm's own weight argument (scalar-mode
    # AdaLN only — per-token conditioning has a (B, N, dim) scale and cannot),
    # and plain mx.compile.
    for tokens in args.tokens:
        h = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
        scale = mx.random.normal((1, 1, dim)).astype(mx.bfloat16)
        shift = mx.random.normal((1, 1, dim)).astype(mx.bfloat16)
        mx.eval(h, scale, shift)
        w = (1.0 + scale).reshape(-1).astype(mx.bfloat16)
        mx.eval(w)

        def naive(v, scale=scale, shift=shift):
            return mx.fast.rms_norm(v, weight=None, eps=eps) * (1.0 + scale) + shift

        def folded(v, w=w, shift=shift):
            return mx.fast.rms_norm(v, weight=w, eps=eps) + shift

        compiled = mx.compile(naive)
        ex_fold = "bit-exact" if _bitexact(naive(h), folded(h)) else f"rel {_rel(folded(h), naive(h)):.2e}"
        ex_comp = "bit-exact" if _bitexact(naive(h), compiled(h)) else f"rel {_rel(compiled(h), naive(h)):.2e}"
        n_s = _time(naive, h, args.iters, args.warmup, 48)
        report("C adaln modulate", tokens, n_s, _time(folded, h, args.iters, args.warmup, 48),
               "rms_norm(weight=1+s)", ex_fold)
        report("C adaln modulate", tokens, n_s, _time(compiled, h, args.iters, args.warmup, 48),
               "mx.compile", ex_comp)

    # ---------------- D — the whole block: the fusion ceiling ---------------
    if not args.skip_block:
        for tokens in args.tokens:
            lf = 16
            lh, lw = (9, 16) if tokens == 2304 else (18, 32)
            if lf * lh * lw != tokens:  # a custom token count: fabricate a grid
                lh, lw = 1, tokens // lf
            n_audio, n_text = 126, 1024
            block = BasicAVTransformerBlock(
                video_dim=dim, audio_dim=adim,
                video_num_heads=cfg.video_num_heads, audio_num_heads=cfg.audio_num_heads,
                video_head_dim=cfg.video_head_dim, audio_head_dim=cfg.audio_head_dim,
                av_cross_num_heads=cfg.av_cross_num_heads, av_cross_head_dim=cfg.av_cross_head_dim,
                ff_mult=cfg.ff_mult, norm_eps=eps, ff_bias=cfg.ff_bias, audio_ff_bias=cfg.audio_ff_bias,
            )
            block.set_dtype(mx.bfloat16)
            nn.quantize(block, group_size=args.group_size, bits=args.bits)
            mx.eval(block.parameters())

            vh = mx.random.normal((1, tokens, dim)).astype(mx.bfloat16)
            ah = mx.random.normal((1, n_audio, adim)).astype(mx.bfloat16)
            v_text = mx.random.normal((1, n_text, dim)).astype(mx.bfloat16)
            a_text = mx.random.normal((1, n_text, adim)).astype(mx.bfloat16)
            v_adaln = mx.random.normal((1, 9 * dim)).astype(mx.bfloat16)
            a_adaln = mx.random.normal((1, 9 * adim)).astype(mx.bfloat16)
            av_v = mx.random.normal((1, 4 * dim)).astype(mx.bfloat16)
            av_a = mx.random.normal((1, 4 * adim)).astype(mx.bfloat16)
            av_vg = mx.random.normal((1, dim)).astype(mx.bfloat16)
            av_ag = mx.random.normal((1, adim)).astype(mx.bfloat16)
            v_pos = compute_video_positions(lf, lh, lw, frame_rate=24.0)
            a_pos = compute_audio_positions(n_audio)
            v_rope = precompute_rope_freqs(v_pos, inner_dim=dim, num_heads=cfg.video_num_heads,
                                           theta=cfg.rope_theta,
                                           max_pos=list(cfg.positional_embedding_max_pos[: v_pos.shape[-1]]),
                                           rope_type=cfg.rope_type)
            a_rope = precompute_rope_freqs(a_pos, inner_dim=adim, num_heads=cfg.audio_num_heads,
                                           theta=cfg.rope_theta,
                                           max_pos=list(cfg.audio_positional_embedding_max_pos),
                                           rope_type=cfg.rope_type)
            cross_max = max(cfg.positional_embedding_max_pos[0], cfg.audio_positional_embedding_max_pos[0])
            vc_rope = precompute_rope_freqs(v_pos[:, :, 0:1],
                                            inner_dim=cfg.av_cross_num_heads * cfg.av_cross_head_dim,
                                            num_heads=cfg.av_cross_num_heads, theta=cfg.rope_theta,
                                            max_pos=[cross_max], rope_type=cfg.rope_type)
            ac_rope = precompute_rope_freqs(a_pos[:, :, 0:1],
                                            inner_dim=cfg.av_cross_num_heads * cfg.av_cross_head_dim,
                                            num_heads=cfg.av_cross_num_heads, theta=cfg.rope_theta,
                                            max_pos=[cross_max], rope_type=cfg.rope_type)
            mx.eval(vh, ah, v_text, a_text)

            def run(pair, block=block):
                v, a = pair
                return block(v, a, v_adaln, a_adaln, None, None, av_v, av_a, av_vg, av_ag,
                             video_text_embeds=v_text, audio_text_embeds=a_text,
                             video_rope_freqs=v_rope, audio_rope_freqs=a_rope,
                             video_cross_rope_freqs=vc_rope, audio_cross_rope_freqs=ac_rope)

            # How many full-width modulate sites does a real block actually
            # have? Counted, not assumed — arm C's per-site number is only
            # meaningful multiplied by a measured count.
            real_rms = mx.fast.rms_norm
            seen: list[tuple] = []

            def counting_rms(x, weight, eps, **kw):
                seen.append((tuple(x.shape), weight is None))
                return real_rms(x, weight, eps, **kw)

            mx.fast.rms_norm = counting_rms  # type: ignore[assignment]
            try:
                mx.eval(run((vh, ah)))
            finally:
                mx.fast.rms_norm = real_rms  # type: ignore[assignment]
            video_sites = sum(1 for s, w in seen if s[1] == tokens and w)
            audio_sites = sum(1 for s, w in seen if s[1] == n_audio and w)
            print(f"   [rms_norm sites in one real block: {video_sites} at video width, "
                  f"{audio_sites} at audio width, {len(seen)} total]")

            compiled_block = mx.compile(run)
            ov, _ = run((vh, ah))
            cv, _ = compiled_block((vh, ah))
            mx.eval(ov, cv)
            exact = "bit-exact" if _bitexact(ov, cv) else f"rel {_rel(cv, ov):.2e}"
            n_s = _time(run, (vh, ah), args.iters, args.warmup, 2)
            f_s = _time(compiled_block, (vh, ah), args.iters, args.warmup, 2)
            report("D whole block (FUSION CEILING)", tokens, n_s, f_s, "mx.compile", exact)

            # Peak memory, both arms. The H3 audit's fused Metal kernel was
            # bit-exact and free of speed but cost +1.87 GiB of peak; a fusion
            # experiment that does not report peak is half a measurement.
            peak_fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory  # type: ignore[attr-defined]
            reset_fn = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)  # type: ignore[attr-defined]
            peaks = {}
            for label, fn in (("eager", run), ("compiled", compiled_block)):
                mx.synchronize()
                if reset_fn is not None:
                    reset_fn()
                mx.eval(fn((vh, ah)))
                mx.synchronize()
                peaks[label] = peak_fn()
            delta = peaks["compiled"] - peaks["eager"]
            print(f"   [peak: eager {peaks['eager'] / 1024**3:.3f} GiB · compiled "
                  f"{peaks['compiled'] / 1024**3:.3f} GiB · delta {delta / 1024**2:+.1f} MiB]")
            rows.append({"site": "D peak memory", "tokens": tokens, "variant": "mx.compile",
                         "eager_peak_bytes": peaks["eager"], "compiled_peak_bytes": peaks["compiled"],
                         "delta_bytes": delta, "saved_pct": 0.0})

    best = max((r for r in rows if not r["site"].startswith("I")), key=lambda r: r["saved_pct"], default=None)
    print()
    if best is not None:
        print(f"best non-instrument arm: {best['site']} @ {best['tokens']} tokens via {best['variant']} "
              f"→ {best['saved_pct']:+.2f}%  (gate: >{args.gate}%)")
        print("VERDICT: " + ("proceed to the end-to-end proof" if best["saved_pct"] > args.gate
                             else "STOP — no site clears the gate; a render-level win cannot exceed the layer-level one"))
    if args.json:
        Path(args.json).write_text(json.dumps({"mlx": mx.__version__, "bits": args.bits,
                                               "group_size": args.group_size, "rows": rows}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
