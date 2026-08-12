# Pipelines & options matrix

Reference for all CLI subcommands of `ltx-2-mlx`, the pipeline class
backing each, and which memory / performance flags apply where.
Current as of **v0.12.1**.

For the underlying architecture and conventions, see
[CLAUDE.md](../CLAUDE.md). For the high-level user-facing overview,
see [README.md](../README.md). For production-readiness classification
of each pipeline (Stable / Beta / Experimental) and stability guarantees
by tier, see [PIPELINE_MATURITY.md](PIPELINE_MATURITY.md).

## Core pipelines

| CLI | Pipeline class | Mode(s) | Sampler stage 1 | Sampler stage 2 | Default model | CFG | STG default |
|---|---|---|---|---|---|---|---|
| `generate --one-stage` | `TI2VidOneStagePipeline` | T2V / I2V | Euler + CFG (30 steps) at **full** resolution | — | q8 + dev LoRA | ✅ | 0.0 |
| `generate --two-stage` | `TI2VidTwoStagesPipeline` | T2V / I2V | Euler + CFG (30 steps) | Euler distilled (3 steps) | q8 + dev LoRA | ✅ | 0.0 |
| `generate --two-stages-hq` | `TI2VidTwoStagesHQPipeline` | T2V / I2V | res_2s + CFG (15 steps × 2 sub-steps) | Euler distilled (3) | q8 + dev LoRA | ✅ | 0.0 |
| `generate --distilled` | `DistilledPipeline` | T2V / I2V | Euler distilled (8 steps) at half-res | Euler distilled (**2** on LTX-2.5, 3 on 2.3) at full-res | q8 (distilled only) | ❌ | — |
| `a2v` *(beta)* | `A2VidPipelineTwoStage` | A2V (+ optional I2V) | Euler + CFG (30) | Euler distilled (3) | q8 + dev LoRA | ✅ (audio cfg=7) | 0.0 |
| `keyframe` | `KeyframeInterpolationPipeline` | start frame ↔ end frame | Euler + CFG (30) | Euler distilled (3) | q8 + dev LoRA | ✅ | 0.0 |
| `ic-lora` | `ICLoraPipeline` | V2V (control video) + optional I2V | Euler distilled (8) | Euler distilled (3) | q8 + control LoRA | ❌ | — |
| `hdr-ic-lora` | `HDRICLoraPipeline(ICLoraPipeline)` | V2V / pure T2V / +I2V → linear HDR | Euler distilled (8) | Euler distilled (3) | q8 + HDR LoRA | ❌ | — |
| `lipdub` *(exp.)* | `LipDubPipeline(ICLoraPipeline)` | V2V + ref audio → lip-dubbed | Euler distilled (8) | Euler distilled (3) | q8 + LipDub LoRA | ❌ | — |
| `retake` *(beta)* | `RetakePipeline.retake_from_video` | regenerate latent frame range | Euler dev + CFG (30) | — | dev | ✅ | 0.0 |
| `extend` *(beta)* | `RetakePipeline.extend_from_video` (same class) | append frames before/after | Euler dev + CFG (30) | — | dev | ✅ | 0.0 |
| `enhance` | Gemma rewrite | prompt → enriched prompt | — | — | Gemma 3 12B | — | — |
| `info` / `train` / `preprocess` | — | utilities | — | — | — | — | — |

## Memory / perf opt-ins (cross-pipeline)

| Flag / env var | Default | Effect | Supported pipelines |
|---|---|---|---|
| `--low-ram` | off | Block streaming: stream DiT layers from mmap'd safetensors. Peak ≈ 1 block + Gemma. ~75% transformer RAM cut. | `generate` (one-stage / `--two-stage` / `--two-stages-hq`), `a2v`, `keyframe`, `ic-lora`, `hdr-ic-lora` |
| `--tile-frames N` | 1 | Split video tokens into N temporal tiles. Caps O(N²) attention activations. | `generate` (all variants), `a2v`, `keyframe` |
| `--tile-spatial M` | 1 | Split video tokens into M×M spatial tiles. Total tiles = `tile-frames × M²`. | same as above |
| `--tile-overlap K` | 2 | Token-grid overlap (smoother blend at cost of redundant compute). | when tiling active |
| `--enable-teacache` | off | Timestep-aware residual caching. ~1.46× speedup (Euler) / ~1.78× (HQ). Conservative thresh 0.5. | `generate --two-stage`, `generate --two-stages-hq` |
| `--teacache-thresh F` | 0.5 (Euler) / 1.0 (HQ) | Skip aggressiveness. Higher = more skip = faster but quality risk. | with `--enable-teacache` |
| `LTX2_GEMMA_EVAL_EVERY=N` | 1 (per-layer eval) | Per-layer `mx.eval` cadence in Gemma forward. Keeps each Metal command buffer below the macOS GPU watchdog (~10 s) deadline. Default `1` on all Apple Silicon since PR #3 (M2 Max 64 GB also crashes without it at production resolutions). Set to `0` on Mac Studio / M-series Ultra owners who never see the watchdog crash to recover lazy-graph throughput. | all pipelines (text encoding shared) |
| `LTX2_DIT_EVAL_EVERY=N` | 8 | Flush the DiT block loop every N blocks (splits 48 blocks into 6 command buffers, ~1-2 s each). Same trade-off as `LTX2_GEMMA_EVAL_EVERY`. Set to `0` to disable. | all pipelines (DiT forward shared) |
| `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` | unset | **AGX driver knob, not an LTX2 variable — never set automatically.** Relaxes the macOS GPU-watchdog eviction timeout for the process that sets it. Mitigates the macOS 26.x + MLX 0.31.x regression where sustained GPU work is killed with `Impacting Interactivity` while the display is active ([ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267)); the CLI prints this guidance when it detects that failure. Trade-off: UI responsiveness may degrade during the run, and on some machines it is not sufficient — running with the display off is the only workaround reported 100% reliable upstream. Unrelated to the ~10 s command-buffer deadline handled by `LTX2_GEMMA_EVAL_EVERY`. | all pipelines (any sustained GPU work) |
| `LTX2_GEMMA_MAX_LENGTH=N` | 1024 | Cap Gemma padded seq_len (last-resort escape hatch). Quality risk: shifts left-padded RoPE positions. | all pipelines |

## Pipeline-specific options

| Pipeline | Specific flags |
|---|---|
| `generate --one-stage` | `--frame-rate` (required), `--stage1-steps` aliased to `num_steps` (default 30), `--cfg-scale` (3.0), `--stg-scale` (0.0), `--image`. No stage2 / TeaCache / distilled-lora flags. Common: `--lora PATH STRENGTH` (incompatible with `--low-ram`), `--enhance-prompt`. |
| `generate --two-stage` | `--frame-rate` (required), `--stage1-steps` (30), `--stage2-steps` (3), `--cfg-scale` (3.0), `--stg-scale` (0.0), `--image`, `--distilled-lora-strength` (1.0), `--enable-teacache`, `--teacache-thresh` |
| `generate --two-stages-hq` | same as two-stage but stage1 default 15 steps, res_2s sampler |
| `generate --distilled` | `--frame-rate` (required), `--stage1-steps` (8 default), `--stage2-steps` (**2 default on LTX-2.5**, 3 on 2.3), `--schedule-preset {default,fast,vendor}`, `--stage1-sigmas` / `--stage2-sigmas` (explicit comma-separated schedules), `--image`. No CFG/STG/TeaCache flags (distilled flow). Same DiT in both stages — no LoRA swap. A step count **thins** the schedule keeping its terminal 0.0; it truncated (leaving an unfinished denoise) before 2026-08-12. See [the distilled schedule](#the-distilled-lanes-sigma-schedule). |
| `a2v` | `--audio` (required), `--image`, `--audio-start`, `--frame-rate` (required), all two-stage flags |
| `keyframe` | `--start` / `--end` (image paths, required), `--frame-rate` (required), all two-stage flags |
| `ic-lora` | `--frame-rate` (required), `--lora PATH STRENGTH` (required, repeatable), `--video-conditioning PATH STRENGTH` (required, repeatable), `--conditioning-strength` (1.0), `--image`, `--skip-stage-2`, `--stage1-steps`, `--stage2-steps` |
| `hdr-ic-lora` | same as `ic-lora`, but `--video-conditioning` is **optional** (omit for pure T2V HDR). Auto-detects HDR transform from LoRA metadata. Outputs `.mp4` SDR + `.hdr.npz` linear HDR fp32 |
| `lipdub` *(experimental)* | `--reference-video PATH` (required, provides visuals + audio), `--lora PATH STRENGTH` (exactly one LipDub IC-LoRA, required), `--reference-strength` (1.0), `--stage1-steps` / `--stage2-steps`. Frame count auto-derived from the reference video metadata (snapped to `8k+1`). Output audio is VAE+vocoder reconstruction — remux original audio for music-fidelity use cases. |
| `retake` | `--video` (required), `--start` / `--end` (latent frame indices, required), `--steps` (30), `--no-regen-audio`, `--cfg-scale`, `--stg-scale` |
| `extend` | `--video` (required), `--extend-frames N` (required), `--direction before|after`, `--steps`, `--cfg-scale`, `--stg-scale` |

## Compatibility notes

- **`--lora-mode {auto,unfused,fuse}`** applies to every `generate` mode. `auto` (default) applies the LoRA as an **unfused runtime branch** on a quantized pack and fuses it into the weights on a float one. Fusing into a quantized weight destroys **~95 %** of a rank-32 delta at q4 (~10 % at q8) — the documented cause of character LoRAs "not triggering". `fuse` is kept as the honest A/B control and now measures + reports what it costs. `--lora-mode unfused` is **not available with `--low-ram`** and raises rather than silently fusing.
- `generate --lora <path>` (one-stage) is **incompatible with `--low-ram`** (LoRA pre-fuse happens before streaming setup). Use `ic-lora` or pre-fuse via mlx-forge.
- `--low-ram` + custom `--distilled-lora-strength` (≠1.0) on two-stage uses bind-time LoRA fusion (slower per step but supports any strength). At strength=1.0, swaps to pre-fused `transformer-distilled.safetensors`.
- TeaCache calibration is sampler-specific (Euler vs res_2s). Don't reuse coefficients across `--two-stage` and `--two-stages-hq`.
- HDR LoRA can be combined with regular IC-LoRA control LoRAs in theory but untested — single HDR LoRA per pipeline is the validated path.
- Modality tiling overhead dominates over memory benefit at default Nv (1650-3168). Use only when targeting 1080p / 8s+ on Mac Studio 64-128 GB; on 32 GB Mac, prefer `--low-ram` alone.
- `generate` requires a mode flag (`--one-stage`, `--two-stage`, `--two-stages-hq`, or `--distilled`). There is **no implicit default** — every pipeline maps 1:1 to an upstream Lightricks/LTX-2 class.
- `generate --one-stage` vs `generate --two-stage`: same dev model + CFG, but `--one-stage` runs **once at the target resolution** (no upscaler dependency, simpler latents for downstream). `--two-stage` runs at half-res then upscales 2× and refines (typically faster overall and better at large targets). Pick `--one-stage` for native res ≤ 480×704 or if you don't trust the upsampler; pick `--two-stage` for everything else.
- `generate --distilled` vs `generate --two-stage`: same half-res + upscale structure, but `--distilled` skips CFG entirely (8 stage 1 steps × 1 forward instead of 30 × 2-4). Fastest mode; quality slightly below the dev+CFG variants.

## The distilled lane's sigma schedule

`generate --distilled` is the only lane whose schedule is a **fixed table** rather
than a computed one, so it is the only lane where "how many steps" and "which
sigmas" are separate questions. Both stages resolve through
`scheduler.resolve_distilled_schedule`, keyed on the checkpoint's generation.

**Every schedule this lane can produce terminates at sigma 0.0.** Before
2026-08-12 a step count *sliced* the table, so `--stage2-steps 2` on LTX-2.5 ran
`[0.85, 0.725, 0.421875]` — an unfinished refine that hands on a latent with
residual noise while reporting itself as "2 steps". A step count now **thins**
the table (both endpoints kept, interior points dropped at a uniform stride) and
a count the table cannot supply is refused with a message.

A step count addresses **the checkpoint's own schedule** — the `vendor` preset —
not whichever preset you picked. So `--stage2-steps 3` on LTX-2.5 still returns
the vendor 3-step list even though the default holds 2, and every existing caller
that passes the old step counts keeps working.

### Defaults, per checkpoint generation

| Generation | stage 1 | stage 2 | forwards |
|---|---|---|---:|
| LTX-2.5 | the vendor's 9 points | `0.85, 0.421875, 0.0` | 8 + 2 = **10** |
| LTX-2.3 (and any unreadable version) | the vendor's 9 points | `0.909375, 0.725, 0.421875, 0.0` | 8 + 3 = 11 |

The distilled lane costs exactly `(len(sigmas₁) − 1) + (len(sigmas₂) − 1)` DiT
forwards — no guider, no substep, one forward per step — so every sigma point
removed is one forward removed.

### `--schedule-preset` (LTX-2.5)

| Preset | Schedule | Wall vs `vendor` | What it does to the output |
|---|---|---:|---|
| `default` | 8 + 2 | **−17 %** | Same take (composition correlation 0.9988), one fewer refine step |
| `fast` | 5 + 2 | **−29 %** | **A different take** (0.920) — a coherent shot, not a cheaper copy of the same one. Drafts and take exploration |
| `vendor` | 8 + 3 | — | The vendor template's own lists; this lane's default before 2026-08-12 |

On LTX-2.3 only `default` / `vendor` exist, both the 2.3 vendor schedule. The
thinned presets were graded on the 2.5 distilled checkpoint and are refused
elsewhere rather than silently remapped.

Measured at 1024×576×121, q8, seed 774411, n=1: vendor **170.2 s**, default
**140.7 s**, fast **120.8 s**. A stage-2 forward costs **4.6×** a stage-1 forward
on this lane (29.6 s vs 6.40 s), because stage 2 runs at full resolution — which
is why stage 2 is the cheapest place to thin.

### `--stage1-sigmas` / `--stage2-sigmas`

Explicit comma-separated schedules, for anything the presets do not cover:

```bash
ltx-2-mlx generate --distilled -p "..." --frame-rate 24 \
  --stage1-sigmas 1.0,0.99375,0.9875,0.98125,0.975,0.0 \
  --stage2-sigmas 0.85,0.421875,0.0 -o out.mp4
```

Validated before anything loads: at least 2 points, strictly decreasing, starting
at or below 1.0, terminating at 0.0, and at most the distilled checkpoint's 9
points. A list and a step count for the same stage is a contradiction and raises.

**Stage 2's first sigma is not just a starting point** — it is the level the
upscaled latent is re-noised to (`noise·σ + stage1·(1−σ)`). At LTX-2.5's 0.85
only **15 %** of stage 1 survives into the refine, which is why stage 1's tail is
cheap to thin and its front is not: the front decides the composition, the tail
is 85 % overwritten. Moving that first value changes far more than one step.
