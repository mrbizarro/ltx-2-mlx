"""The ancestral sampler is not just implemented — it is REACHED.

`tests/test_ltx25_ancestral_sampler.py` proves ``EulerAncestralDiffusionStep``
computes the right numbers. It passed for weeks while **nothing instantiated
the class**, so every LTX-2.5 render this project produced ran the LTX-2.3
Euler step. A correctness test on an unreachable code path is worth nothing,
and that is the specific failure this file exists to make impossible.

So the assertions here are about *wiring and selection*, not arithmetic:

* a 2.3 checkpoint still takes the byte-identical Euler path;
* a 2.5 checkpoint does not;
* ``denoise_loop`` actually applies the stepper it is handed — asserted by
  computing the expected update longhand and by showing the two samplers
  disagree at a fixed seed;
* the version is found through every wrapper a pipeline might have put around
  the DiT, because a version lookup that silently fails would re-create the
  original bug while looking wired.

It also pins the LTX-2.5 STG / modality-guidance position (HF Diffusers PR
"stg_scale and modality_scale default to the SFT values and are gated
independently of guidance_scale") — see the final section.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mlx.core as mx
import pytest
from mlx_arsenal.diffusion import euler_step

from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core_mlx.components.guiders import MultiModalGuiderParams
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.model.transformer.model import LTXModelConfig
from ltx_pipelines_mlx.utils.sampler_choice import (
    ANCESTRAL_ETA,
    ANCESTRAL_SAMPLER_SINCE_VERSION,
    ANCESTRAL_S_NOISE,
    KEYFRAME_ETA,
    SAMPLER_ENV_VAR,
    model_version_of,
    resolve_diffusion_step,
)
from ltx_pipelines_mlx.utils.samplers import denoise_loop


# --------------------------------------------------------------------------- #
# doubles — no weights, no GPU
# --------------------------------------------------------------------------- #


@dataclass
class FakeDit:
    """Just enough of an LTXModel: it carries a config with a version."""

    config: LTXModelConfig


@dataclass
class Wrapper:
    """Stands in for X0Model / StreamingLTXModel / TiledLTXModel, all of which
    hold the real model on ``.model``."""

    model: object


def dit(version: tuple[int, int]) -> FakeDit:
    return FakeDit(config=LTXModelConfig(model_version=version))


X0_GAIN = 0.5


class LinearX0:
    """An X0Model that is a fixed, deterministic function of its input.

    ``x0 = X0_GAIN * x``. Deterministic, so any divergence between two runs is
    the sampler and nothing else — and *input-dependent*, which a constant x0
    is not. That distinction is load-bearing: both samplers return the denoised
    prediction verbatim when ``sigma_next == 0``, so with a constant x0 every
    schedule ending at 0.0 collapses to the same final value and a
    "the samplers differ" test passes vacuously. (Written the wrong way first;
    the test caught it, which is the only reason this docstring exists.)
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, *, video_latent, audio_latent, **_kwargs):
        self.calls += 1
        return X0_GAIN * video_latent, X0_GAIN * audio_latent


def state(shape=(1, 4, 8), fill: float = 1.0) -> LatentState:
    latent = mx.full(shape, fill, dtype=mx.float32)
    return LatentState(
        latent=latent,
        clean_latent=mx.zeros(shape, dtype=mx.float32),
        denoise_mask=mx.ones((shape[0], shape[1], 1), dtype=mx.float32),
    )


SIGMAS = [1.0, 0.6, 0.25, 0.0]


def run_loop(diffusion_step, seed: int = 4242):
    mx.random.seed(seed)
    out = denoise_loop(
        model=LinearX0(),
        video_state=state(),
        audio_state=state((1, 3, 8), fill=0.5),
        video_text_embeds=mx.zeros((1, 2, 4)),
        audio_text_embeds=mx.zeros((1, 2, 4)),
        sigmas=SIGMAS,
        show_progress=False,
        diffusion_step=diffusion_step,
    )
    mx.eval(out.video_latent, out.audio_latent)
    return out


@pytest.fixture(autouse=True)
def _clear_override():
    """No test may leak the env override into the next one."""
    os.environ.pop(SAMPLER_ENV_VAR, None)
    yield
    os.environ.pop(SAMPLER_ENV_VAR, None)


# --------------------------------------------------------------------------- #
# 1. finding the version at all
# --------------------------------------------------------------------------- #


def test_version_is_found_through_every_wrapper_a_pipeline_may_add():
    """X0Model, streaming and tiling wrappers all nest on ``.model``. A lookup
    that stopped at the first wrapper would report "no version" and silently
    hand 2.5 the legacy sampler — the original bug, wearing a fix."""
    bare = dit((2, 5))
    assert model_version_of(bare) == (2, 5)
    assert model_version_of(Wrapper(bare)) == (2, 5)
    assert model_version_of(Wrapper(Wrapper(Wrapper(bare)))) == (2, 5)


def test_unknown_model_yields_no_version_and_therefore_the_legacy_step():
    assert model_version_of(None) == ()
    assert model_version_of(object()) == ()
    assert () < ANCESTRAL_SAMPLER_SINCE_VERSION
    assert resolve_diffusion_step(object()) is None


def test_a_cycle_or_deep_nest_cannot_hang_the_lookup():
    node = Wrapper(None)
    node.model = node  # self-referential wrapper
    assert model_version_of(node) == ()


# --------------------------------------------------------------------------- #
# 2. selection
# --------------------------------------------------------------------------- #


def test_ltx23_selects_the_legacy_euler_path():
    """None is the contract: it means "do not change the 2.3 code path"."""
    assert resolve_diffusion_step(dit((2, 3))) is None


def test_ltx25_selects_the_ancestral_step_at_the_template_eta():
    step = resolve_diffusion_step(dit((2, 5)))
    assert isinstance(step, EulerAncestralDiffusionStep)
    assert step.eta == ANCESTRAL_ETA == 1.0
    assert step.s_noise == ANCESTRAL_S_NOISE == 1.0


def test_a_future_generation_keeps_the_ancestral_step():
    assert isinstance(resolve_diffusion_step(dit((2, 6))), EulerAncestralDiffusionStep)
    assert isinstance(resolve_diffusion_step(dit((3, 0))), EulerAncestralDiffusionStep)


def test_keyframe_eta_resolves_to_the_legacy_path_on_25():
    """The official flf2v template pins eta=0, where the ancestral step is
    arithmetically the Euler step. Returning None rather than a zero-eta
    stepper keeps keyframe interpolation on the byte-identical path and skips
    a per-step noise tensor that would be multiplied by zero."""
    assert KEYFRAME_ETA == 0.0
    assert resolve_diffusion_step(dit((2, 5)), eta=KEYFRAME_ETA) is None


# --------------------------------------------------------------------------- #
# 3. the override
# --------------------------------------------------------------------------- #


def test_override_can_force_either_sampler_regardless_of_version():
    os.environ[SAMPLER_ENV_VAR] = "euler"
    assert resolve_diffusion_step(dit((2, 5))) is None
    os.environ[SAMPLER_ENV_VAR] = "euler_ancestral"
    assert isinstance(resolve_diffusion_step(dit((2, 3))), EulerAncestralDiffusionStep)


def test_a_misspelled_override_raises_rather_than_being_ignored():
    """Silently ignoring it would make an A/B measure nothing while looking
    like it measured something."""
    os.environ[SAMPLER_ENV_VAR] = "ancestrall"
    with pytest.raises(ValueError, match="not a sampler"):
        resolve_diffusion_step(dit((2, 5)))


# --------------------------------------------------------------------------- #
# 4. the loop actually applies what it is handed
# --------------------------------------------------------------------------- #


def test_no_stepper_reproduces_the_euler_update_exactly():
    """The 2.3 regression guard, computed longhand rather than snapshotted."""
    out = run_loop(None)
    x = mx.full((1, 4, 8), 1.0, dtype=mx.float32)
    for sigma, sigma_next in zip(SIGMAS[:-1], SIGMAS[1:]):
        x = euler_step(x, X0_GAIN * x, sigma, sigma_next)
    mx.eval(x)
    assert mx.allclose(out.video_latent, x, atol=0, rtol=0), "the Euler path moved"


def test_the_ancestral_stepper_changes_the_result_at_the_same_seed():
    """If this ever passes trivially — i.e. the two are equal — the stepper is
    not being applied and the wiring has regressed to dead code."""
    euler_out = run_loop(None)
    anc_out = run_loop(EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0))
    assert not mx.allclose(euler_out.video_latent, anc_out.video_latent, atol=1e-6)
    assert not mx.allclose(euler_out.audio_latent, anc_out.audio_latent, atol=1e-6)


def test_the_ancestral_run_is_reproducible_at_a_fixed_seed():
    """Ancestral sampling is stochastic per step but seeded once per run, so
    an A/B at one seed still differs only by the sampler."""
    a = run_loop(EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0), seed=99)
    b = run_loop(EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0), seed=99)
    assert mx.allclose(a.video_latent, b.video_latent, atol=0, rtol=0)
    assert mx.allclose(a.audio_latent, b.audio_latent, atol=0, rtol=0)


def test_a_zero_eta_stepper_matches_the_euler_path_through_the_loop():
    """End-to-end confirmation of the identity the flf2v template relies on."""
    euler_out = run_loop(None)
    zero_out = run_loop(EulerAncestralDiffusionStep(eta=0.0, s_noise=1.0))
    assert mx.allclose(euler_out.video_latent, zero_out.video_latent, atol=1e-6)


def test_video_and_audio_draw_independent_noise():
    """Sharing one draw across modalities would correlate their stochastic
    components for no reason. Asserted on the noise the stepper actually
    receives rather than inferred from the output, because the output is also
    shaped by the schedule and could hide a shared draw."""
    seen: list[mx.array] = []

    class Recording(EulerAncestralDiffusionStep):
        def step(self, sample, denoised_sample, sigmas, step_index, noise=None, **kw):
            seen.append(noise)
            return super().step(sample, denoised_sample, sigmas, step_index, noise=noise, **kw)

    run_loop(Recording(eta=1.0, s_noise=1.0))
    assert len(seen) == 2 * (len(SIGMAS) - 1), "one draw per modality per step"
    video_noise, audio_noise = seen[0], seen[1]
    assert video_noise.shape != audio_noise.shape or not mx.allclose(video_noise, audio_noise)
    # and consecutive steps must not reuse a draw either
    assert not mx.allclose(seen[0], seen[2], atol=1e-6)


# --------------------------------------------------------------------------- #
# 5. the STG / modality-guidance position (HF Diffusers fix)
# --------------------------------------------------------------------------- #


def test_guidance_scales_default_to_neutral_not_to_the_sft_values():
    """Upstream Diffusers shipped snippets where ``stg_scale`` and
    ``modality_scale`` defaulted to the dev/SFT values and were gated
    INDEPENDENTLY of ``guidance_scale`` — so ``guidance_scale=1`` did not turn
    them off and the distilled examples silently ran STG + modality guidance.
    HF staff merged the fix; both official references now agree that distilled
    means no STG and no modality guidance.

    Our defaults are already the corrected ones. This pins that, because the
    repo docs quote the SFT numbers (LTX_2_3_PARAMS: stg 1.0, modality 3.0) as
    'reference params', and a well-meaning edit could promote them to
    defaults."""
    params = MultiModalGuiderParams()
    assert params.stg_scale == 0.0, "STG must be OFF unless explicitly asked for"
    assert params.modality_scale == 1.0, "1.0 is the neutral value; 3.0 is the SFT value"


def test_the_distilled_lane_cannot_run_guidance_at_all():
    """The stronger guarantee, and the reason we were never exposed: the
    distilled pipeline never builds a guider, and ``denoise_loop`` takes none.
    STG on distilled is not merely off by default here — it is unreachable."""
    import inspect

    from ltx_pipelines_mlx import distilled

    source = inspect.getsource(distilled)
    for knob in ("stg_scale", "modality_scale", "guider_factory"):
        assert knob not in source, f"the distilled lane grew a {knob} — re-read the vendor fix"

    assert "video_guider_factory" not in inspect.signature(denoise_loop).parameters


def test_no_scale_is_resolved_with_a_falsy_or():
    """The second half of the vendor trap: ``audio_stg_scale or stg_scale``
    resolves a legitimate 0.0 back to the video value. Grepped rather than
    imagined — if someone adds an audio-side scale, this catches the idiom."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "packages"
    offenders = []
    pattern = re.compile(r"\b\w*(?:scale|strength|eta)\w*\s+or\s+\w", re.IGNORECASE)
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line) and not line.lstrip().startswith(("#", '"', "'")):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "falsy-or on a scale that can legitimately be 0.0:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# 6. the stage-2 schedule, pinned to the official template
# --------------------------------------------------------------------------- #


def test_stage1_sigmas_are_the_official_template_values_verbatim():
    """Node 404 (`ManualSigmas`) of Comfy-Org/workflow_templates
    `video_ltx2_5_t2v.json`, read from the raw JSON rather than from a summary.

    Five of the nine points sit inside the top 2.5% of the noise range. No
    naive shifted schedule reproduces that, so this is a table to copy, not a
    formula to re-derive — and a table that is copied must be pinned."""
    from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS

    assert DISTILLED_SIGMAS == [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    assert sum(1 for s in DISTILLED_SIGMAS if s >= 0.975) == 5


def test_stage2_sigmas_differ_between_generations_in_exactly_one_place():
    """Node 395 of the same template: "0.85, 0.7250, 0.4219, 0.0".

    0.4219 is 0.421875 shown at four decimals — display rounding, not a
    different sigma. The FIRST value is the real change (0.909375 -> 0.85), and
    the fact that it is the only one is what makes it deliberate."""
    from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, STAGE_2_SIGMAS_LTX25

    assert STAGE_2_SIGMAS_LTX25 == [0.85, 0.725, 0.421875, 0.0]
    assert STAGE_2_SIGMAS == [0.909375, 0.725, 0.421875, 0.0]
    differing = [i for i, (a, b) in enumerate(zip(STAGE_2_SIGMAS, STAGE_2_SIGMAS_LTX25)) if a != b]
    assert differing == [0]
    assert round(STAGE_2_SIGMAS_LTX25[2], 4) == 0.4219


def test_stage2_resolution_is_version_keyed_and_defaults_to_23():
    from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, STAGE_2_SIGMAS_LTX25, resolve_stage2_sigmas

    assert resolve_stage2_sigmas((2, 3)) == STAGE_2_SIGMAS
    assert resolve_stage2_sigmas((2, 5)) == STAGE_2_SIGMAS_LTX25
    assert resolve_stage2_sigmas((2, 6)) == STAGE_2_SIGMAS_LTX25
    # an unreadable checkpoint yields (), which must NOT opt into the new table
    assert resolve_stage2_sigmas(()) == STAGE_2_SIGMAS


def test_stage2_truncation_keeps_the_pre_existing_semantics():
    """The old call site was ``STAGE_2_SIGMAS[: n + 1] if n else STAGE_2_SIGMAS``,
    including the wart that a truncated schedule no longer ends at 0.0. Changing
    that here would be a second behaviour change smuggled in under the first."""
    from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, resolve_stage2_sigmas

    for n in (None, 0, 1, 2, 3, 9):
        expected = STAGE_2_SIGMAS[: n + 1] if n else STAGE_2_SIGMAS
        assert resolve_stage2_sigmas((2, 3), n) == expected
    assert resolve_stage2_sigmas((2, 5), 2) == [0.85, 0.725, 0.421875]  # no terminal 0.0, as before


def test_audio_rate_25_is_tokens_per_second_not_a_video_frame_rate():
    """Resolves the reported "docs say 24 fps, the template's audio latent
    implies 25" discrepancy: the 25 is AUDIO LATENT TOKENS PER SECOND, derived
    as 16000 / 160 / 4, and it is independent of the video frame rate. There is
    no fps disagreement to reconcile — we ship 24 fps like 2.3, and the audio
    token count for a 24 fps clip is not 24-shaped."""
    from ltx_core_mlx.utils.positions import AUDIO_LATENTS_PER_SECOND, compute_audio_token_count

    assert AUDIO_LATENTS_PER_SECOND == 25.0
    # the official template's base latent is 97 frames; at 24 fps that is not 97 tokens
    assert compute_audio_token_count(97, frame_rate=24.0) == 101
    assert compute_audio_token_count(97, frame_rate=25.0) == 97


# --------------------------------------------------------------------------- #
# 7. IC-LoRAs are distilled-only on 2.5 (vendor guidance)
# --------------------------------------------------------------------------- #


def test_ic_lora_pipeline_keeps_all_its_methods():
    """Guard against a class-body edit accident, not against a feature.

    The 2.5 warning below was first inserted between two methods at column 0,
    which silently ENDED the class: every method after it became a module-level
    function, `py_compile` was perfectly happy, and ICLoraPipeline lost half its
    API. Cheap to assert, and the failure is invisible to a syntax check."""
    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline

    for method in ("load", "_effective_lora_paths", "_fuse_loras", "generate_and_save"):
        assert callable(getattr(ICLoraPipeline, method, None)), f"ICLoraPipeline lost {method}"


def test_ic_lora_on_a_25_dev_checkpoint_warns_but_does_not_refuse(caplog):
    """Lightricks documents the 2.5 IC-LoRAs as distilled-only ("do not use with
    the dev checkpoint"). Our dev-mode path is the ComfyUI Union-Control recipe
    for 2.3, which is validated — 2.5 changes the advice, not the mechanics.

    A warning rather than a refusal is the deliberate choice: refusing would
    remove a working feature from anyone whose panel routes an IC-LoRA through
    the High tier (the HDR IC-LoRA does exactly that), and that trade is a
    product decision. The vendor's failure mode is degraded output, not a
    crash, so the job here is to make it attributable."""
    import logging

    from ltx_pipelines_mlx.ic_lora import _warn_if_ic_lora_on_a_25_dev_checkpoint as warn

    with caplog.at_level(logging.WARNING, logger="ltx_pipelines_mlx.ic_lora"):
        warn(FakeDit(config=LTXModelConfig(model_version=(2, 5))), 2)
    assert "distilled-only" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ltx_pipelines_mlx.ic_lora"):
        warn(FakeDit(config=LTXModelConfig(model_version=(2, 3))), 2)
    assert caplog.text == "", "2.3's Union-Control recipe is validated; it must stay silent"


def test_the_ic_lora_warning_can_never_break_a_render():
    """It runs on the hot path of a real generation. Anything it touches that
    is unexpected must be swallowed, not raised."""
    from ltx_pipelines_mlx.ic_lora import _warn_if_ic_lora_on_a_25_dev_checkpoint as warn

    for junk in (None, object(), 0, "not a model"):
        warn(junk, 1)


# --------------------------------------------------------------------------- #
# 8. isolated-modality guidance: OFF by default on 2.5, untouched on 2.3
# --------------------------------------------------------------------------- #
#
# This is the one section in this file that pins a **deliberate output change**
# rather than an invariance. Isolated-modality guidance is a real guidance term;
# turning it off changes the picture. It is off on 2.5 because it was measured
# and then graded by eye, not because it was proved neutral:
#
#   arm G_modality_off, 1024x576x121, seed 774411, LTX-2.5 q8 dev + distilled
#   LoRA 450: 306.9 s -> 246.2 s (-60.7 s, -19.8 %), peak footprint unmoved
#   (39.52 vs 39.53 GB). Owner verdict 2026-08-12: "G modality is nice" (PASS).
#   The sibling arm that buys the same 61 s by dropping CFG instead was FAILED
#   ("D2 changes character and has visual weirdness") — hence CFG is untouched
#   here and only the modality pass goes.
#
# Evidence: ~/AI/projects/phosphene/notes/ltx25_perf_exp1.md (arm G_modality_off)
# and board row 1 of ltx25_perf_board.md.
#
# The 2.3 assertions below are the load-bearing half. TI2VidTwoStagesHQPipeline
# is generation-agnostic — `ltx-2-mlx generate --two-stages-hq` against a 2.3
# checkpoint reaches this exact code — so a default that was not version-keyed
# would have silently changed 2.3 as well.


def test_modality_guidance_is_off_by_default_on_25_and_sft_on_23():
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import (
        MODALITY_SCALE_NEUTRAL,
        MODALITY_SCALE_SFT,
        resolve_modality_scale,
    )

    assert MODALITY_SCALE_SFT == 3.0
    assert MODALITY_SCALE_NEUTRAL == 1.0

    assert resolve_modality_scale((2, 5)) == MODALITY_SCALE_NEUTRAL
    assert resolve_modality_scale((2, 6)) == MODALITY_SCALE_NEUTRAL
    assert resolve_modality_scale((3, 0)) == MODALITY_SCALE_NEUTRAL

    # 2.3 and anything older keeps the SFT value it shipped with.
    assert resolve_modality_scale((2, 3)) == MODALITY_SCALE_SFT
    assert resolve_modality_scale((2, 0)) == MODALITY_SCALE_SFT
    # an unreadable checkpoint yields (), which must NOT opt into the change
    assert resolve_modality_scale(()) == MODALITY_SCALE_SFT


def test_the_23_hq_defaults_are_byte_for_byte_what_they_always_were():
    """The regression that would matter most: keying the default by generation
    but getting the comparison backwards, or letting it leak onto 2.3.

    `LTX_2_3_HQ_PARAMS` is the reference table this path has always built, so
    it is compared field by field rather than on modality_scale alone — a
    version-keyed edit that also nudged rescale_scale or the audio CFG would
    pass a modality-only assertion."""
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import build_hq_guider_params
    from ltx_pipelines_mlx.utils.constants import LTX_2_3_HQ_PARAMS

    video, audio = build_hq_guider_params((2, 3), cfg_scale=3.0, stg_scale=0.0)

    pairs = (
        (video, LTX_2_3_HQ_PARAMS.video_guider_params),
        (audio, LTX_2_3_HQ_PARAMS.audio_guider_params),
    )
    for built, reference in pairs:
        assert built.cfg_scale == reference.cfg_scale
        assert built.stg_scale == reference.stg_scale
        assert built.stg_blocks == reference.stg_blocks
        assert built.rescale_scale == reference.rescale_scale
        assert built.modality_scale == reference.modality_scale == 3.0
        assert built.skip_step == reference.skip_step


def test_only_the_modality_scale_moves_between_23_and_25():
    """The change is one field wide. Asserting *that* is what stops a future
    edit from riding along on the same version key."""
    from dataclasses import fields

    from ltx_pipelines_mlx.ti2vid_two_stages_hq import build_hq_guider_params

    v23, a23 = build_hq_guider_params((2, 3), cfg_scale=3.0, stg_scale=0.0)
    v25, a25 = build_hq_guider_params((2, 5), cfg_scale=3.0, stg_scale=0.0)

    for old, new in ((v23, v25), (a23, a25)):
        differing = [f.name for f in fields(old) if getattr(old, f.name) != getattr(new, f.name)]
        assert differing == ["modality_scale"], differing
        assert old.modality_scale == 3.0 and new.modality_scale == 1.0


def test_a_25_default_actually_switches_the_isolated_modality_pass_off():
    """The number is only a saving if the guider stops asking for the pass.

    `_predict` builds the isolated-modality forward iff
    `do_isolated_modality_generation()` is true on either guider, so that
    predicate — not the float — is what the 61 seconds are made of."""
    from ltx_core_mlx.components.guiders import MultiModalGuider
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import build_hq_guider_params

    v25, a25 = build_hq_guider_params((2, 5), cfg_scale=3.0, stg_scale=0.0)
    assert not MultiModalGuider(params=v25).do_isolated_modality_generation()
    assert not MultiModalGuider(params=a25).do_isolated_modality_generation()

    # ...and CFG is deliberately still on, because arm D2 was FAILED by the owner.
    assert MultiModalGuider(params=v25).do_unconditional_generation()
    assert MultiModalGuider(params=a25).do_unconditional_generation()

    v23, a23 = build_hq_guider_params((2, 3), cfg_scale=3.0, stg_scale=0.0)
    assert MultiModalGuider(params=v23).do_isolated_modality_generation()
    assert MultiModalGuider(params=a23).do_isolated_modality_generation()


def test_a_caller_supplied_guider_params_still_wins_on_both_generations():
    """The escape hatch is the pre-existing one: pass your own params. If the
    new default overrode them, turning modality guidance back on for a 2.5
    render would be impossible from outside the library."""
    from ltx_pipelines_mlx.ti2vid_two_stages_hq import build_hq_guider_params

    mine = MultiModalGuiderParams(cfg_scale=3.0, stg_scale=0.0, rescale_scale=0.45, modality_scale=3.0, stg_blocks=[])

    for version in ((2, 3), (2, 5)):
        video, audio = build_hq_guider_params(
            version,
            cfg_scale=3.0,
            stg_scale=0.0,
            video_guider_params=mine,
            audio_guider_params=mine,
        )
        assert video is mine and audio is mine
        assert video.modality_scale == 3.0

    # ...and half an override leaves the other side on the resolved default
    video, audio = build_hq_guider_params((2, 5), cfg_scale=3.0, stg_scale=0.0, video_guider_params=mine)
    assert video is mine and video.modality_scale == 3.0
    assert audio is not mine and audio.modality_scale == 1.0


def test_the_hq_pipeline_reaches_the_resolver_rather_than_a_literal():
    """The failure this file exists to prevent: a correct default nobody calls.

    `generate_two_stage` needs 26 GB of weights, so the call site is asserted by
    source rather than executed — but it is asserted as an ABSENCE too: a bare
    `modality_scale=3.0` literal anywhere in this module would mean the resolver
    was added beside the old code instead of replacing it."""
    import inspect

    from ltx_pipelines_mlx import ti2vid_two_stages_hq as hq

    source = inspect.getsource(hq.TI2VidTwoStagesHQPipeline.generate_two_stage)
    assert "build_hq_guider_params(" in source
    assert "model_version_of(self.dit)" in source
    assert "modality_scale=3.0" not in source

    body = inspect.getsource(hq.build_hq_guider_params)
    assert "resolve_modality_scale(" in body
    assert "modality_scale=3.0" not in body


def test_the_other_pipelines_keep_their_sft_modality_scale():
    """Scope guard. The owner passed this arm on the two-stage HQ path, which is
    the only path experiment 1 measured and the only path he graded. The Euler
    two-stage, one-stage, a2v and retake pipelines are NOT covered by that
    verdict and must still ship the SFT value."""
    import inspect

    from ltx_pipelines_mlx import a2vid_two_stage, retake, ti2vid_one_stage, ti2vid_two_stages

    for module in (ti2vid_one_stage, ti2vid_two_stages, retake, a2vid_two_stage):
        assert "modality_scale=3.0" in inspect.getsource(module), module.__name__
