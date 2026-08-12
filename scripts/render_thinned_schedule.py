#!/usr/bin/env python3
"""Render the distilled lane on an EXPLICIT sigma schedule, and count its forwards.

Written for ``CODEX_EXPERIMENTS_LTX25.md`` Experiment 5 (thinning the distilled
schedule). The point of this script is the trap the brief names:

    ``sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1]``

Truncating with ``--stage1-steps`` means the schedule **no longer ends at
0.0**, so ``--stage1-steps 6`` is not "6 steps", it is an unfinished denoise
with residual noise left in the latent. Any A/B built on it measures the wrong
thing. So arms here do not truncate: they pass a full, explicit, re-spaced list
that still terminates at 0.0, and the list used is recorded verbatim in the
JSON beside the wall time.

Everything is a **script-level monkeypatch**. The library is untouched, a normal
render is byte-for-byte unchanged, and deleting this file has no consequence.
Two patches, both narrow:

* ``ltx_pipelines_mlx.distilled.DISTILLED_SIGMAS`` — stage 1's schedule.
* ``ltx_pipelines_mlx.distilled.resolve_stage2_sigmas`` — stage 2's.

plus a wrapper on ``DistilledPipeline.generate_two_stage`` that drops
``stage1_steps`` / ``stage2_steps`` when the matching override is present, so
the pipeline takes the whole overridden list instead of re-slicing it and
re-introducing the very truncation this script exists to avoid.

The forward counter is the same instrument Experiment 1 used
(``scripts/count_dit_forwards.py``): it wraps ``X0Model.__call__``, records only
shapes and the scalar sigma, and forces no extra ``mx.eval``. On the distilled
lane there is no guider and no second-order substep, so the expected count is
simply ``len(sigmas) - 1`` per stage — but the brief's own rule is to *count*
rather than derive it, so it is counted.

Usage::

    python scripts/render_thinned_schedule.py \\
        --name front6 \\
        --stage1-sigmas 1.0,0.975,0.909375,0.725,0.421875,0.0 \\
        --json out/front6.json -- \\
        generate --distilled --model ... -o front6.mp4 ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def _parse_sigmas(raw: str) -> list[float]:
    values = [float(v) for v in raw.replace(" ", "").split(",") if v != ""]
    if len(values) < 2:
        raise SystemExit(f"--*-sigmas needs at least 2 points, got {values}")
    if values[-1] != 0.0:
        raise SystemExit(
            f"schedule must terminate at 0.0 or it leaves residual noise "
            f"(that is the whole point of this script): got {values}"
        )
    for a, b in zip(values[:-1], values[1:]):
        if not a > b:
            raise SystemExit(f"schedule must be strictly decreasing: {values}")
    if values[0] > 1.0:
        raise SystemExit(f"first sigma above 1.0 is out of the noise range: {values}")
    return values


def main() -> int:
    argv = sys.argv[1:]
    name = "arm"
    json_out: Path | None = None
    stage1: list[float] | None = None
    stage2: list[float] | None = None

    while argv and argv[0].startswith("--"):
        flag = argv[0]
        if flag == "--":
            argv = argv[1:]
            break
        if len(argv) < 2:
            raise SystemExit(f"{flag} needs a value")
        value = argv[1]
        argv = argv[2:]
        if flag == "--name":
            name = value
        elif flag == "--json":
            json_out = Path(value)
        elif flag == "--stage1-sigmas":
            stage1 = _parse_sigmas(value)
        elif flag == "--stage2-sigmas":
            stage2 = _parse_sigmas(value)
        else:
            raise SystemExit(f"unknown flag {flag}")
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2

    import mlx.core as mx

    from ltx_core_mlx.model.transformer.model import X0Model
    from ltx_pipelines_mlx import distilled as _distilled

    baseline_stage1 = list(_distilled.DISTILLED_SIGMAS)

    # --- schedule overrides -------------------------------------------------
    if stage1 is not None:
        if len(stage1) > len(baseline_stage1):
            raise SystemExit(
                f"stage 1 has {len(stage1)} points; the distilled checkpoint's schedule holds "
                f"{len(baseline_stage1)} ({len(baseline_stage1) - 1} steps) and the pipeline "
                f"silently clamps above it. Thinning only."
            )
        _distilled.DISTILLED_SIGMAS = stage1
    if stage2 is not None:
        _distilled.resolve_stage2_sigmas = lambda _version, _steps=None, _s=stage2: list(_s)

    if stage1 is not None or stage2 is not None:
        real_generate = _distilled.DistilledPipeline.generate_two_stage

        def generate_without_truncation(self, *a, **kw):
            # A step count would re-slice the overridden list and put the
            # non-terminating tail back. Drop it, loudly.
            if stage1 is not None and kw.pop("stage1_steps", None) is not None:
                print("[thinned] ignoring --stage1-steps: an explicit stage-1 schedule was given")
            if stage2 is not None and kw.pop("stage2_steps", None) is not None:
                print("[thinned] ignoring --stage2-steps: an explicit stage-2 schedule was given")
            return real_generate(self, *a, **kw)

        _distilled.DistilledPipeline.generate_two_stage = generate_without_truncation

    # --- forward counter ----------------------------------------------------
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

    from ltx_pipelines_mlx import cli

    sys.argv = ["ltx-2-mlx", *argv]
    status = 0
    try:
        cli.main()
    except SystemExit as exc:
        status = int(exc.code or 0)
    finally:
        wall = time.perf_counter() - t0
        X0Model.__call__ = original_call
        payload = {
            "name": name,
            "mlx_version": mx.__version__,
            "argv": argv,
            "stage1_sigmas": stage1 if stage1 is not None else baseline_stage1,
            "stage2_sigmas": stage2,
            "stage1_overridden": stage1 is not None,
            "stage2_overridden": stage2 is not None,
            "sampler_env": os.environ.get("LTX_SAMPLER"),
            "wall_seconds": round(wall, 2),
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "summary": _summarize(records),
            "forwards": records,
        }
        text = json.dumps(payload, indent=2)
        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(text)
            print(f"\n[thinned] wrote {json_out}")
        print("\n[thinned] " + json.dumps(payload["summary"]))
    return status


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"total_forwards": 0}
    token_counts = sorted({r["video_tokens"] for r in records})
    stage1_tokens = token_counts[0]

    def bucket(name: str, rows: list[dict]) -> dict:
        return {
            "name": name,
            "forwards": len(rows),
            "video_tokens": rows[0]["video_tokens"] if rows else None,
            "sigmas_seen": [r["sigma"] for r in rows],
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
