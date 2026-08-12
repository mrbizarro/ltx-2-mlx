#!/usr/bin/env python3
"""Count the DiT forwards a render actually performs, then time it.

Written for CODEX_EXPERIMENTS_LTX25.md Experiment 1: *count* the stage-1
forwards instead of deriving them from ``steps x 2 x 2``.

The counter is a **script-level monkeypatch** of ``X0Model.__call__``. Nothing
in the library changes, so a normal render is byte-for-byte untouched and this
file can be deleted without consequence. Each call records only shapes and the
scalar sigma -- never a full tensor -- so no extra ``mx.eval`` is forced and the
lazy graph keeps its normal shape. Entry timestamps are therefore *dispatch*
times, not per-forward compute times; use the per-stage tqdm ``s/it`` for
timing and the counts here for arithmetic.

Passes are distinguishable without touching the sampler:

* ``block_stack_override is not None`` -> this forward was a **TeaCache skip**
  (prelude + head only, the 48-block stack replaced by a cached residual).
* ``perturbations is not None``        -> STG or modality-isolated pass.
* video token count                    -> stage 1 (half res) vs stage 2 (full res).

Optional arms, all env-gated and all off by default:

* ``LTX_EXP1_MODALITY_SCALE``  override the HQ pipeline's ``modality_scale``
  (it ships at the SFT value 3.0, which makes the isolated-modality pass run).
* ``LTX_EXP1_AUDIO_CFG``       override the HQ pipeline's audio ``cfg_scale``
  (it ships at 7.0, which keeps the unconditional pass alive even when the
  video CFG is 1.0 -- ``_predict`` ORs the two guiders).
* ``LTX_EXP1_TAP=1``           attach a calibration tap to the stage-1 loop and
  record per-step (delta_in, delta_out) so TeaCache coefficients can be refit
  on real 2.5 dynamics. Note this makes the sampler compute the gate signal on
  every step even with TeaCache off, exactly as the TeaCache path does.

Usage::

    python scripts/count_dit_forwards.py --json out.json -- \\
        generate --two-stages-hq --model ... -o arm_a.mp4 ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    return None if raw is None or raw == "" else float(raw)


def main() -> int:
    argv = sys.argv[1:]
    json_out = None
    if argv and argv[0] == "--json":
        json_out = Path(argv[1])
        argv = argv[2:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2

    import mlx.core as mx

    from ltx_core_mlx.model.transformer.model import X0Model

    records: list[dict] = []
    t0 = time.perf_counter()

    original_call = X0Model.__call__

    def counting_call(self, video_latent, audio_latent, sigma, *args, **kwargs):
        records.append(
            {
                "i": len(records),
                "t": round(time.perf_counter() - t0, 4),
                "video_tokens": int(video_latent.shape[1]),
                "audio_tokens": int(audio_latent.shape[1]),
                "sigma": round(float(sigma[0].item()), 6),
                "perturbed": kwargs.get("perturbations") is not None,
                "teacache_skip": kwargs.get("block_stack_override") is not None,
            }
        )
        return original_call(self, video_latent, audio_latent, sigma, *args, **kwargs)

    X0Model.__call__ = counting_call

    # --- optional guider overrides (arms G / D2) ---------------------------
    modality_override = _env_float("LTX_EXP1_MODALITY_SCALE")
    audio_cfg_override = _env_float("LTX_EXP1_AUDIO_CFG")
    if modality_override is not None or audio_cfg_override is not None:
        from ltx_core_mlx.components import guiders as _guiders
        from ltx_pipelines_mlx import ti2vid_two_stages_hq as _hq

        real_params = _guiders.MultiModalGuiderParams

        def patched_params(**kw):
            # The HQ pipeline builds the video params with rescale_scale 0.45
            # and the audio params with 1.0; that is how the two are told
            # apart without reaching into the pipeline.
            is_audio = abs(float(kw.get("rescale_scale", 0.0)) - 1.0) < 1e-9
            if modality_override is not None:
                kw["modality_scale"] = modality_override
            if audio_cfg_override is not None and is_audio:
                kw["cfg_scale"] = audio_cfg_override
            return real_params(**kw)

        _hq.MultiModalGuiderParams = patched_params

    # --- optional calibration tap (partial arm F) --------------------------
    tap_rows: list[dict] = []
    if os.environ.get("LTX_EXP1_TAP") == "1":
        from ltx_pipelines_mlx import ti2vid_two_stages_hq as _hq

        real_loop = _hq.res2s_denoise_loop
        state: dict = {"gate": None, "vres": None, "ares": None}

        def rel_l1(curr, prev) -> float:
            c = curr.astype(mx.float32)
            p = prev.astype(mx.float32)
            return float((mx.mean(mx.abs(c - p)) / mx.mean(mx.abs(p))).item())

        def calibration_tap(step_idx, gate, v_res, a_res):
            row = {"step": int(step_idx)}
            if state["gate"] is not None:
                row["delta_in"] = rel_l1(gate, state["gate"])
                row["delta_out_video"] = rel_l1(v_res, state["vres"])
                row["delta_out_audio"] = rel_l1(a_res, state["ares"])
            state["gate"], state["vres"], state["ares"] = gate, v_res, a_res
            tap_rows.append(row)

        def tapped_loop(*a, **kw):
            kw.setdefault("tap", None)
            if kw["tap"] is None:
                kw["tap"] = calibration_tap
            return real_loop(*a, **kw)

        _hq.res2s_denoise_loop = tapped_loop

    from ltx_pipelines_mlx import cli

    sys.argv = ["ltx-2-mlx", *argv]
    status = 0
    try:
        cli.main()
    except SystemExit as exc:  # argparse / CLI exits
        status = int(exc.code or 0)
    finally:
        wall = time.perf_counter() - t0
        X0Model.__call__ = original_call
        payload = {
            "mlx_version": mx.__version__,
            "argv": argv,
            "wall_seconds": round(wall, 2),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "env_overrides": {
                k: v for k, v in os.environ.items() if k.startswith(("LTX_EXP1_", "LTX2_", "LTX_"))
            },
            "summary": _summarize(records),
            "forwards": records,
            "teacache_calibration": tap_rows,
        }
        text = json.dumps(payload, indent=2)
        if json_out is not None:
            json_out.write_text(text)
            print(f"\n[count_dit_forwards] wrote {json_out}")
        print("\n[count_dit_forwards] " + json.dumps(payload["summary"]))
    return status


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"total_forwards": 0}
    token_counts = sorted({r["video_tokens"] for r in records})
    stage1_tokens = token_counts[0]

    def bucket(name: str, rows: list[dict]) -> dict:
        return {
            "forwards": len(rows),
            "full": sum(1 for r in rows if not r["teacache_skip"]),
            "teacache_skipped": sum(1 for r in rows if r["teacache_skip"]),
            "perturbed": sum(1 for r in rows if r["perturbed"]),
            "video_tokens": rows[0]["video_tokens"] if rows else None,
            "distinct_sigmas": len({r["sigma"] for r in rows}),
            "name": name,
        }

    s1 = [r for r in records if r["video_tokens"] == stage1_tokens]
    s2 = [r for r in records if r["video_tokens"] != stage1_tokens]
    return {
        "total_forwards": len(records),
        "video_token_counts": token_counts,
        "stage1": bucket("stage1", s1),
        "stage2": bucket("stage2", s2),
    }


if __name__ == "__main__":
    raise SystemExit(main())
