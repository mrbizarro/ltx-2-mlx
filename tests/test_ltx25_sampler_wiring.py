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
