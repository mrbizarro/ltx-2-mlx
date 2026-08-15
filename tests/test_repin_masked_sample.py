"""Contract: the masked-sample re-pin holds an i2v anchor through both
stochastic loops, and turning it off (Inspire) is a deliberate choice.

The +ltx25.4 fix re-composited the conditioned tokens into the SAMPLE after
the euler-ancestral step. The res_2s loop had the same defect class — SDE
noise injected unmasked at both substep and step level — and was left for an
owner decision; the owner's field report ("high-quality generation that is
different from the reference") is that decision arriving. +ltx25.5 closes it:
``repin_masked_sample`` on both loops, resolved per generation by the HQ
pipeline, with ``loose_reference`` (Inspire) as the sanctioned opt-out.

Schedules here deliberately do NOT end at 0.0: the terminal denoise stamps
the clean latent back through the x0 mask on every path, which is exactly
the masking effect that made the original bug invisible — the delivered
frame 0 matched while the trajectory (and therefore the composition) never
saw the anchor. Ending mid-schedule exposes the sample itself.
"""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.components.diffusion_steps import EulerAncestralDiffusionStep
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_pipelines_mlx.utils.samplers import denoise_loop, res2s_denoise_loop


X0_GAIN = 0.5
ANCHOR = 7.0
SIGMAS_OPEN = [1.0, 0.6, 0.25]  # no terminal 0.0 — see the module docstring


class LinearX0:
    def __call__(self, *, video_latent, audio_latent, **_kwargs):
        return X0_GAIN * video_latent, X0_GAIN * audio_latent


def masked_state(shape=(1, 4, 8)) -> LatentState:
    """First token conditioned (mask 0, clean = ANCHOR), rest generated.

    The conditioned token STARTS at the clean value, as every real pipeline
    initializes it — that precondition is what makes plain Euler's velocity
    zero at a pinned token. (The first draft of this fixture started it at
    the generated fill and 'proved' Euler drifts; the test caught the wrong
    assumption, which is why this docstring exists.)"""
    latent = mx.ones(shape, dtype=mx.float32)
    latent[:, 0, :] = ANCHOR
    clean = mx.full(shape, ANCHOR, dtype=mx.float32)
    mask = mx.ones((shape[0], shape[1], 1), dtype=mx.float32)
    mask[:, 0, :] = 0.0
    return LatentState(latent=latent, clean_latent=clean, denoise_mask=mask)


def _anchor_token(out) -> mx.array:
    return out.video_latent[:, 0, :]


def _run_res2s(repin: bool):
    mx.random.seed(99)
    out = res2s_denoise_loop(
        model=LinearX0(),
        video_state=masked_state(),
        audio_state=masked_state((1, 3, 8)),
        video_text_embeds=mx.zeros((1, 2, 4)),
        audio_text_embeds=mx.zeros((1, 2, 4)),
        sigmas=list(SIGMAS_OPEN),
        show_progress=False,
        bongmath=False,
        repin_masked_sample=repin,
    )
    mx.eval(out.video_latent, out.audio_latent)
    return out


def _run_euler_ancestral(repin: bool):
    mx.random.seed(99)
    out = denoise_loop(
        model=LinearX0(),
        video_state=masked_state(),
        audio_state=masked_state((1, 3, 8)),
        video_text_embeds=mx.zeros((1, 2, 4)),
        audio_text_embeds=mx.zeros((1, 2, 4)),
        sigmas=list(SIGMAS_OPEN),
        show_progress=False,
        diffusion_step=EulerAncestralDiffusionStep(eta=1.0),
        repin_masked_sample=repin,
    )
    mx.eval(out.video_latent, out.audio_latent)
    return out


def test_res2s_repin_holds_the_anchor():
    held = _anchor_token(_run_res2s(repin=True))
    assert bool(mx.allclose(held, mx.full(held.shape, ANCHOR)).item()), (
        "with repin on, the conditioned token must remain the clean latent "
        "through the SDE updates"
    )


def test_res2s_without_repin_buries_the_anchor():
    drifted = _anchor_token(_run_res2s(repin=False))
    assert not bool(mx.allclose(drifted, mx.full(drifted.shape, ANCHOR)).item()), (
        "without repin the SDE noise must visibly move the conditioned token "
        "— if this ever passes, the defect this file documents has silently "
        "changed shape"
    )


def test_ancestral_repin_flag_matches_the_shipped_fix():
    held = _anchor_token(_run_euler_ancestral(repin=True))
    assert bool(mx.allclose(held, mx.full(held.shape, ANCHOR)).item())
    drifted = _anchor_token(_run_euler_ancestral(repin=False))
    assert not bool(mx.allclose(drifted, mx.full(drifted.shape, ANCHOR)).item()), (
        "repin=False (Inspire) must restore the pre-fix drift on purpose"
    )


def test_plain_euler_ignores_the_flag_entirely():
    def run(repin: bool):
        mx.random.seed(7)
        out = denoise_loop(
            model=LinearX0(),
            video_state=masked_state(),
            audio_state=masked_state((1, 3, 8)),
            video_text_embeds=mx.zeros((1, 2, 4)),
            audio_text_embeds=mx.zeros((1, 2, 4)),
            sigmas=list(SIGMAS_OPEN),
            show_progress=False,
            diffusion_step=None,
            repin_masked_sample=repin,
        )
        mx.eval(out.video_latent, out.audio_latent)
        return out

    a, b = run(True), run(False)
    assert bool(mx.allclose(a.video_latent, b.video_latent).item()), (
        "Euler is analytically anchor-stable; the flag must not change a byte"
    )
    assert bool(mx.allclose(a.audio_latent, b.audio_latent).item())
