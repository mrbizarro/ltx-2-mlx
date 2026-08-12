#!/usr/bin/env python3
"""Measure the attention K/V memory share of one LTX-2.5 denoise forward.

Experiment 7 of ``CODEX_EXPERIMENTS_LTX25.md``. The question is narrow and the
budget is 20 minutes: **does the LTX attention retain K/V across a forward, and
if so how many GiB is the live pair at peak?** Below 5 % of the phase peak the
experiment is dropped without a line of kernel being written — which is what
happened on H3 (0.68 %).

Method, and why it is not a full render:

* K/V bytes are a function of *shapes*, not of weight values. One real
  ``BasicAVTransformerBlock`` built from the real checkpoint header, run at the
  real token counts, produces exactly the K/V tensors the 48-block DiT produces
  — 48 times over, one block at a time. Random weights change no shape.
* The instrument is a wrapper around ``mx.fast.scaled_dot_product_attention``,
  so it sees the *actual* arrays handed to attention rather than an
  algebraic reconstruction of them.
* Retention is decided by reading the code and then *checking* it: the module's
  own parameter/state dict is snapshotted before and after the forward.

Denominator: the pinned High-tier render's measured peak (``ltx25_dev_pack.md``
§6B.2 — 18.88 GB RSS / 39.58 GB footprint at 1024x576x121), passed in with
``--phase-peak-gb`` so the number in the report is never a guess.

    python3 scripts/measure_kv_share.py --stage 2
    python3 scripts/measure_kv_share.py --stage 1 --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.model.transformer.model import LTXModelConfig
from ltx_core_mlx.model.transformer.rope import precompute_rope_freqs
from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

#: The six attention calls a block makes, in execution order. Used to name the
#: SDPA calls the instrument intercepts — the wrapper sees tensors, not modules.
SITE_ORDER = (
    "attn1 (video self)",
    "audio_attn1 (audio self)",
    "attn2 (video x text)",
    "audio_attn2 (audio x text)",
    "audio_to_video_attn (A2V)",
    "video_to_audio_attn (V2A)",
)

GB = 1024**3
MB = 1024**2


def _peak_bytes() -> int:
    fn = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory  # type: ignore[attr-defined]
    return int(fn())


def _reset_peak() -> None:
    fn = getattr(mx, "reset_peak_memory", None) or getattr(mx.metal, "reset_peak_memory", None)  # type: ignore[attr-defined]
    if fn is not None:
        fn()


def _active_bytes() -> int:
    fn = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory  # type: ignore[attr-defined]
    return int(fn())


def _latent_grid(width: int, height: int, frames: int) -> tuple[int, int, int]:
    """(F, H, W) latent grid for a pixel canvas. VAE is 8x temporal, 32x spatial."""
    return ((frames - 1) // 8 + 1, height // 32, width // 32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=2, help="1 = half-res stage 1, 2 = full res")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--frame-rate", type=float, default=24.0)
    ap.add_argument("--text-tokens", type=int, default=1024, help="Gemma padded length (LTX2_GEMMA_MAX_LENGTH)")
    ap.add_argument(
        "--checkpoint",
        default="/Users/salo/pinokio/api/phosphene-dev.git/mlx_models/ltx-2.5-mlx-q8/transformer-dev.safetensors",
        help="Read the real architecture out of this checkpoint's header (no weights loaded)",
    )
    ap.add_argument("--phase-peak-gb", type=float, default=18.88, help="Measured render peak RSS, GB (the denominator)")
    ap.add_argument("--footprint-gb", type=float, default=39.58, help="Measured render footprint, GB")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # ---- Real architecture, read from the checkpoint's own header ----------
    cfg = LTXModelConfig.from_checkpoint_file(args.checkpoint)
    cfg_source = args.checkpoint
    if cfg is None:
        cfg = LTXModelConfig()
        cfg_source = "DEFAULTS (checkpoint header unreadable — say so in the report)"

    width, height = (args.width, args.height) if args.stage == 2 else (args.width // 2, args.height // 2)
    # Stage 1 runs the half-res canvas floored to a multiple of 32 (the pipeline's own rule).
    width, height = (width // 32) * 32, (height // 32) * 32
    lf, lh, lw = _latent_grid(width, height, args.frames)
    n_video = lf * lh * lw
    n_audio = round(args.frames / args.frame_rate * 25)
    n_text = args.text_tokens

    print(f"mlx {mx.__version__}")
    print(f"config from  {cfg_source}")
    print(f"             model_version={cfg.model_version} layers={cfg.num_layers} "
          f"video {cfg.video_num_heads}x{cfg.video_head_dim}={cfg.video_dim} "
          f"audio {cfg.audio_num_heads}x{cfg.audio_head_dim}={cfg.audio_dim} "
          f"av_cross {cfg.av_cross_num_heads}x{cfg.av_cross_head_dim}")
    print(f"stage {args.stage}: {width}x{height}x{args.frames} -> latent {lf}x{lh}x{lw} "
          f"= {n_video} video tokens, {n_audio} audio, {n_text} text\n")

    # ---- One real block, bf16, random weights (shapes are what matter) -----
    mx.random.seed(0)
    block = BasicAVTransformerBlock(
        video_dim=cfg.video_dim,
        audio_dim=cfg.audio_dim,
        video_num_heads=cfg.video_num_heads,
        audio_num_heads=cfg.audio_num_heads,
        video_head_dim=cfg.video_head_dim,
        audio_head_dim=cfg.audio_head_dim,
        av_cross_num_heads=cfg.av_cross_num_heads,
        av_cross_head_dim=cfg.av_cross_head_dim,
        ff_mult=cfg.ff_mult,
        norm_eps=cfg.norm_eps,
        ff_bias=cfg.ff_bias,
        audio_ff_bias=cfg.audio_ff_bias,
    )
    block.set_dtype(mx.bfloat16)
    mx.eval(block.parameters())

    def state_fingerprint() -> dict[str, tuple]:
        """Every array the module holds, by path — the retention audit's subject."""
        out: dict[str, tuple] = {}

        def walk(prefix: str, obj) -> None:
            if isinstance(obj, mx.array):
                out[prefix] = (obj.shape, obj.dtype.__str__(), obj.nbytes)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    walk(f"{prefix}.{i}", v)

        walk("", dict(block.parameters()))
        walk("state", {k: v for k, v in vars(block).items() if not k.startswith("__")})
        return out

    before = state_fingerprint()

    # ---- Real inputs at the real geometry ---------------------------------
    vh = mx.random.normal((1, n_video, cfg.video_dim)).astype(mx.bfloat16)
    ah = mx.random.normal((1, n_audio, cfg.audio_dim)).astype(mx.bfloat16)
    v_text = mx.random.normal((1, n_text, cfg.video_dim)).astype(mx.bfloat16)
    a_text = mx.random.normal((1, n_text, cfg.audio_dim)).astype(mx.bfloat16)
    v_adaln = mx.random.normal((1, 9 * cfg.video_dim)).astype(mx.bfloat16)
    a_adaln = mx.random.normal((1, 9 * cfg.audio_dim)).astype(mx.bfloat16)
    av_v = mx.random.normal((1, 4 * cfg.video_dim)).astype(mx.bfloat16)
    av_a = mx.random.normal((1, 4 * cfg.audio_dim)).astype(mx.bfloat16)
    av_v_gate = mx.random.normal((1, cfg.video_dim)).astype(mx.bfloat16)
    av_a_gate = mx.random.normal((1, cfg.audio_dim)).astype(mx.bfloat16)

    # Real positions -> real RoPE tables, because they are retained for the
    # whole forward and belong in any honest attention-memory accounting.
    from ltx_core_mlx.utils.positions import compute_audio_positions, compute_video_positions

    v_pos = compute_video_positions(lf, lh, lw, frame_rate=args.frame_rate)
    a_pos = compute_audio_positions(n_audio)
    if v_pos.ndim == 2:
        v_pos = v_pos[None]
    if a_pos.ndim == 2:
        a_pos = a_pos[None]
    v_rope = precompute_rope_freqs(
        v_pos,
        inner_dim=cfg.video_dim,
        num_heads=cfg.video_num_heads,
        theta=cfg.rope_theta,
        max_pos=list(cfg.positional_embedding_max_pos[: v_pos.shape[-1]]),
        rope_type=cfg.rope_type,
    )
    a_rope = precompute_rope_freqs(
        a_pos,
        inner_dim=cfg.audio_dim,
        num_heads=cfg.audio_num_heads,
        theta=cfg.rope_theta,
        max_pos=list(cfg.audio_positional_embedding_max_pos),
        rope_type=cfg.rope_type,
    )
    cross_max = max(cfg.positional_embedding_max_pos[0], cfg.audio_positional_embedding_max_pos[0])
    v_cross_rope = precompute_rope_freqs(
        v_pos[:, :, 0:1],
        inner_dim=cfg.av_cross_num_heads * cfg.av_cross_head_dim,
        num_heads=cfg.av_cross_num_heads,
        theta=cfg.rope_theta,
        max_pos=[cross_max],
        rope_type=cfg.rope_type,
    )
    a_cross_rope = precompute_rope_freqs(
        a_pos[:, :, 0:1],
        inner_dim=cfg.av_cross_num_heads * cfg.av_cross_head_dim,
        num_heads=cfg.av_cross_num_heads,
        theta=cfg.rope_theta,
        max_pos=[cross_max],
        rope_type=cfg.rope_type,
    )

    def rope_bytes(freqs) -> int:
        cos, sin, _ = freqs
        return int(cos.nbytes + sin.nbytes)

    rope_live = rope_bytes(v_rope) + rope_bytes(a_rope) + rope_bytes(v_cross_rope) + rope_bytes(a_cross_rope)
    mx.eval(vh, ah, v_text, a_text, v_rope[0], v_rope[1], a_rope[0], a_rope[1],
            v_cross_rope[0], v_cross_rope[1], a_cross_rope[0], a_cross_rope[1])

    # ---- The instrument: intercept the real SDPA calls ---------------------
    records: list[dict] = []
    real_sdpa = mx.fast.scaled_dot_product_attention

    def recording_sdpa(q, k, v, *pos, **kw):
        idx = len(records)
        records.append(
            {
                "site": SITE_ORDER[idx] if idx < len(SITE_ORDER) else f"site {idx}",
                "q_shape": list(q.shape),
                "kv_shape": list(k.shape),
                "dtype": str(k.dtype),
                "k_bytes": int(k.nbytes),
                "v_bytes": int(v.nbytes),
                "q_bytes": int(q.nbytes),
                "scores_if_materialised": int(q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * 2),
            }
        )
        return real_sdpa(q, k, v, *pos, **kw)

    mx.fast.scaled_dot_product_attention = recording_sdpa  # type: ignore[assignment]
    try:
        mx.synchronize()
        _reset_peak()
        base_active = _active_bytes()
        out_v, out_a = block(
            vh,
            ah,
            v_adaln,
            a_adaln,
            None if not cfg.use_prompt_adaln_single else mx.random.normal((1, 2 * cfg.video_dim)).astype(mx.bfloat16),
            None if not cfg.use_prompt_adaln_single else mx.random.normal((1, 2 * cfg.audio_dim)).astype(mx.bfloat16),
            av_v,
            av_a,
            av_v_gate,
            av_a_gate,
            video_text_embeds=v_text,
            audio_text_embeds=a_text,
            video_rope_freqs=v_rope,
            audio_rope_freqs=a_rope,
            video_cross_rope_freqs=v_cross_rope,
            audio_cross_rope_freqs=a_cross_rope,
        )
        mx.eval(out_v, out_a)
        mx.synchronize()
        block_peak = _peak_bytes()
    finally:
        mx.fast.scaled_dot_product_attention = real_sdpa  # type: ignore[assignment]

    after = state_fingerprint()
    new_state = {k: v for k, v in after.items() if k not in before}
    changed = {k: (before[k], after[k]) for k in before if k in after and before[k] != after[k]}

    # ---- Report ------------------------------------------------------------
    print(f"{'attention site':<28} {'q':>16} {'k/v each':>16} {'K+V MB':>9} {'scores MB*':>11}")
    print("-" * 86)
    per_site_kv = []
    for r in records:
        kv = r["k_bytes"] + r["v_bytes"]
        per_site_kv.append(kv)
        print(
            f"{r['site']:<28} {str(tuple(r['q_shape'])):>16} {str(tuple(r['kv_shape'])):>16} "
            f"{kv / MB:>9.1f} {r['scores_if_materialised'] / MB:>11.1f}"
        )
    print("* scores column = what a NON-flash SDPA would materialise (B·H·Nq·Nk·2B); "
          "reported for context, not measured as live.\n")

    max_live_kv = max(per_site_kv) if per_site_kv else 0
    block_kv_sum = sum(per_site_kv)
    all_blocks_kv = block_kv_sum * cfg.num_layers
    phase_peak = args.phase_peak_gb * 1e9  # GB as reported by /usr/bin/time -l (decimal)

    print(f"attention calls in one block          : {len(records)}")
    print(f"K/V retained across the forward       : {'NO — nothing new in module state' if not new_state else new_state}")
    print(f"module state changed by the forward   : {'none' if not changed else changed}")
    print(f"largest single live K/V pair          : {max_live_kv / MB:>9.1f} MB "
          f"= {max_live_kv / phase_peak * 100:.3f} % of the {args.phase_peak_gb} GB phase peak")
    print(f"all K/V live at once, one block       : {block_kv_sum / MB:>9.1f} MB "
          f"= {block_kv_sum / phase_peak * 100:.3f} % (upper bound: they are not all live)")
    print(f"a FULL cache of all {cfg.num_layers} blocks' K/V   : {all_blocks_kv / GB:>9.2f} GiB "
          f"= {all_blocks_kv / phase_peak * 100:.1f} % — the thing INT8 K/V would compress, "
          f"and it does not exist on this path")
    print(f"RoPE tables live for the whole forward: {rope_live / MB:>9.1f} MB "
          f"= {rope_live / phase_peak * 100:.3f} % (these ARE retained across all blocks)")
    print(f"one instrumented block forward peak   : {block_peak / GB:>9.2f} GiB (mx.get_peak_memory, "
          f"active before = {base_active / MB:.1f} MB)")
    print(f"largest live K/V as a share of that   : {max_live_kv / block_peak * 100:>9.2f} % "
          f"(the most favourable denominator available)")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "mlx": mx.__version__,
                    "config_source": cfg_source,
                    "stage": args.stage,
                    "canvas": [width, height, args.frames],
                    "tokens": {"video": n_video, "audio": n_audio, "text": n_text},
                    "sites": records,
                    "max_live_kv_bytes": max_live_kv,
                    "block_kv_sum_bytes": block_kv_sum,
                    "all_blocks_kv_bytes": all_blocks_kv,
                    "rope_live_bytes": rope_live,
                    "block_forward_peak_bytes": block_peak,
                    "phase_peak_gb": args.phase_peak_gb,
                    "footprint_gb": args.footprint_gb,
                    "kv_retained": bool(new_state),
                    "state_changed": {k: [list(v[0]), list(v[1])] for k, v in changed.items()},
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
