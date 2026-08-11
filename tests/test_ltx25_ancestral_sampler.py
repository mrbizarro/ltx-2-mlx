"""Numeric tests for the LTX-2.5 Euler-ancestral sampler.

No weights are needed to check a sampler: it is closed-form arithmetic on
``(sample, denoised, sigmas, step)``. So this suite does the strongest thing
available without the gated checkpoint — it transcribes upstream's PyTorch
formulas into plain NumPy, independently of the MLX implementation, and
asserts the two agree elementwise.

The reference is ``EulerAncestralDiffusionStep`` from
``ltx-core`` v1.2.0 (``ltx_core/components/diffusion_steps.py``), the sampler
upstream gates at ``ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)``.

The trap this suite exists to catch: the file already contains
``_get_ancestral_step``, the DDIM / variance-exploding ancestral helper used
by the CFG++ step. Reaching for it here — the names match, after all — gives
a different ``sigma_down`` and a different amount of injected noise for the
same ``eta``. The two parameterizations agree ONLY at ``eta=0``, so a wrong
implementation passes any test that forgets to exercise ``eta > 0``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.components.diffusion_steps import (
    EulerAncestralDiffusionStep,
    EulerDiffusionStep,
    _get_ancestral_step,
)

# Upstream ltx_pipelines.utils.constants (v1.2.0). Unchanged from 2.3 —
# 2.5 reuses the distilled schedule and only swaps the stage-1 sampler.
DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]

# Upstream ltx_pipelines.distilled
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0


def reference_ancestral_step(sample, denoised, sigmas, step_index, noise, eta, s_noise):
    """Upstream's step, transcribed to NumPy from the PyTorch source.

    Deliberately written out longhand rather than factored, so it reads as a
    direct transcription and a reviewer can diff it against upstream by eye.
    """
    sigma = float(sigmas[step_index])
    sigma_next = float(sigmas[step_index + 1])
    if sigma_next == 0:
        return denoised.astype(sample.dtype)

    x = sample.astype(np.float64)
    d = denoised.astype(np.float64)

    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio

    sigma_down_ratio = sigma_down / sigma
    x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * d

    if eta > 0:
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_coeff = max(sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, 0.0) ** 0.5
        x_next = (alpha_next / alpha_down) * x_next + noise.astype(np.float64) * s_noise * renoise_coeff
    return x_next


@pytest.fixture
def sigmas():
    return mx.array(DISTILLED_SIGMA_VALUES, dtype=mx.float32)


@pytest.fixture
def tensors():
    rng = np.random.default_rng(20260812)
    sample = rng.standard_normal((2, 7, 16)).astype(np.float32)
    denoised = rng.standard_normal((2, 7, 16)).astype(np.float32)
    noise = rng.standard_normal((2, 7, 16)).astype(np.float32)
    return sample, denoised, noise


class TestAgainstTheReferenceFormulas:
    @pytest.mark.parametrize("step_index", range(len(DISTILLED_SIGMA_VALUES) - 1))
    @pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 1.0])
    def test_matches_upstream_elementwise(self, sigmas, tensors, step_index, eta):
        sample, denoised, noise = tensors
        stepper = EulerAncestralDiffusionStep(eta=eta, s_noise=ANCESTRAL_S_NOISE)

        got = np.array(
            stepper.step(
                mx.array(sample),
                mx.array(denoised),
                sigmas,
                step_index,
                noise=mx.array(noise) if eta > 0 else None,
            )
        )
        want = reference_ancestral_step(
            sample, denoised, DISTILLED_SIGMA_VALUES, step_index, noise, eta, ANCESTRAL_S_NOISE
        )
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)

    @pytest.mark.parametrize("step_index", range(len(STAGE_2_DISTILLED_SIGMA_VALUES) - 1))
    def test_matches_on_the_stage_2_schedule_too(self, tensors, step_index):
        sample, denoised, noise = tensors
        stepper = EulerAncestralDiffusionStep(eta=ANCESTRAL_ETA, s_noise=ANCESTRAL_S_NOISE)
        got = np.array(
            stepper.step(
                mx.array(sample),
                mx.array(denoised),
                mx.array(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=mx.float32),
                step_index,
                noise=mx.array(noise),
            )
        )
        want = reference_ancestral_step(
            sample, denoised, STAGE_2_DISTILLED_SIGMA_VALUES, step_index, noise, ANCESTRAL_ETA, ANCESTRAL_S_NOISE
        )
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)

    def test_s_noise_scales_only_the_injected_term(self, sigmas, tensors):
        sample, denoised, noise = tensors
        a = EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0)
        b = EulerAncestralDiffusionStep(eta=1.0, s_noise=2.0)
        got_a = np.array(a.step(mx.array(sample), mx.array(denoised), sigmas, 2, noise=mx.array(noise)))
        got_b = np.array(b.step(mx.array(sample), mx.array(denoised), sigmas, 2, noise=mx.array(noise)))
        want_b = reference_ancestral_step(sample, denoised, DISTILLED_SIGMA_VALUES, 2, noise, 1.0, 2.0)
        np.testing.assert_allclose(got_b, want_b, rtol=2e-5, atol=2e-5)
        assert not np.allclose(got_a, got_b)


class TestDegenerateCases:
    def test_eta_zero_is_a_plain_euler_step(self, sigmas, tensors):
        """eta=0 collapses to the deterministic sampler — exactly, not roughly."""
        sample, denoised, _ = tensors
        ancestral = EulerAncestralDiffusionStep(eta=0.0)
        euler = EulerDiffusionStep()
        for step_index in range(len(DISTILLED_SIGMA_VALUES) - 2):  # skip the sigma_next == 0 step
            got = np.array(ancestral.step(mx.array(sample), mx.array(denoised), sigmas, step_index))
            want = np.array(euler.step(mx.array(sample), mx.array(denoised), sigmas, step_index))
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)

    def test_final_step_returns_the_denoised_prediction(self, sigmas, tensors):
        """sigma_next == 0: nothing left to renoise into."""
        sample, denoised, noise = tensors
        last = len(DISTILLED_SIGMA_VALUES) - 2
        stepper = EulerAncestralDiffusionStep(eta=1.0)
        got = np.array(stepper.step(mx.array(sample), mx.array(denoised), sigmas, last, noise=mx.array(noise)))
        np.testing.assert_allclose(got, denoised, rtol=0, atol=0)

    def test_noise_is_required_when_eta_is_positive(self, sigmas, tensors):
        """Silently sampling deterministically when asked for SDE is worse
        than failing, so upstream raises and so do we."""
        sample, denoised, _ = tensors
        stepper = EulerAncestralDiffusionStep(eta=1.0)
        with pytest.raises(ValueError, match="noise"):
            stepper.step(mx.array(sample), mx.array(denoised), sigmas, 0)

    def test_dtype_is_preserved(self, sigmas, tensors):
        sample, denoised, noise = tensors
        stepper = EulerAncestralDiffusionStep(eta=1.0)
        out = stepper.step(
            mx.array(sample).astype(mx.bfloat16),
            mx.array(denoised).astype(mx.bfloat16),
            sigmas,
            1,
            noise=mx.array(noise).astype(mx.bfloat16),
        )
        assert out.dtype == mx.bfloat16


class TestNotTheDdimHelper:
    """Guard the exact confusion that would make this look right and be wrong.

    Finding worth writing down, because it makes the trap *harder* to spot:
    at the endpoints the two ``sigma_down`` formulas coincide algebraically.

        DDIM        sigma_up   = sigma_to * sqrt(1 - (sigma_to/sigma_from)^2)
                    sigma_down = sqrt(sigma_to^2 - sigma_up^2) = sigma_to^2 / sigma_from
        rectified   sigma_down = sigma_to * (1 + (sigma_to/sigma_from - 1)) = sigma_to^2 / sigma_from

    So at ``eta=1`` — the value LTX-2.5 actually ships — swapping in the DDIM
    helper for ``sigma_down`` alone would look correct. The step still differs,
    because the rectified-flow version rescales the signal by
    ``alpha_next / alpha_down`` and injects a different ``renoise_coeff``. At
    intermediate ``eta`` even ``sigma_down`` diverges. Both are asserted below;
    testing only ``eta=1``'s ``sigma_down`` would have proved nothing.
    """

    def _ddim_sigma_down(self, sigma_from, sigma_to, eta):
        down, _ = _get_ancestral_step(
            mx.array(sigma_from, dtype=mx.float32), mx.array(sigma_to, dtype=mx.float32), eta=eta
        )
        return float(down)

    def _rectified_sigma_down(self, sigma_from, sigma_to, eta):
        return sigma_to * (1.0 + (sigma_to / sigma_from - 1.0) * eta)

    @pytest.mark.parametrize("eta", [0.25, 0.5, 0.75])
    def test_sigma_down_diverges_at_intermediate_eta(self, eta):
        sigma_from, sigma_to = 0.975, 0.909375
        assert not np.isclose(
            self._ddim_sigma_down(sigma_from, sigma_to, eta),
            self._rectified_sigma_down(sigma_from, sigma_to, eta),
            atol=1e-4,
        ), "DDIM and rectified-flow sigma_down must differ for 0 < eta < 1"

    @pytest.mark.parametrize("eta", [0.0, 1.0])
    def test_sigma_down_coincides_at_the_endpoints(self, eta):
        """Documented on purpose: this coincidence is why the trap is subtle."""
        sigma_from, sigma_to = 0.975, 0.909375
        assert np.isclose(
            self._ddim_sigma_down(sigma_from, sigma_to, eta),
            self._rectified_sigma_down(sigma_from, sigma_to, eta),
            atol=1e-5,
        )

    def test_the_full_step_still_differs_at_eta_one(self, sigmas, tensors):
        """Where the parameterizations really part company: the renoise term."""
        sample, denoised, noise = tensors
        step_index = 4
        sigma = DISTILLED_SIGMA_VALUES[step_index]
        sigma_next = DISTILLED_SIGMA_VALUES[step_index + 1]

        got = np.array(
            EulerAncestralDiffusionStep(eta=1.0).step(
                mx.array(sample), mx.array(denoised), sigmas, step_index, noise=mx.array(noise)
            )
        )

        # A DDIM-style ancestral step over the same sigmas: deterministic move
        # to sigma_down, then add sigma_up * noise with NO alpha rescale.
        down, up = _get_ancestral_step(
            mx.array(sigma, dtype=mx.float32), mx.array(sigma_next, dtype=mx.float32), eta=1.0
        )
        ratio = float(down) / sigma
        ddim_like = ratio * sample + (1.0 - ratio) * denoised + float(up) * noise

        assert not np.allclose(got, ddim_like, rtol=1e-3, atol=1e-3), (
            "The rectified-flow step must not reduce to a DDIM ancestral step "
            "at eta=1 — the alpha_next/alpha_down rescale is load-bearing."
        )


class TestSigmaScheduleIsUnchanged:
    """2.5 reuses 2.3's distilled schedule; only the stage-1 sampler moves."""

    def test_mlx_scheduler_matches_upstream_values(self):
        from ltx_pipelines_mlx import scheduler as sched

        found = []
        for name in dir(sched):
            value = getattr(sched, name)
            if isinstance(value, (list, tuple)) and list(value) == DISTILLED_SIGMA_VALUES:
                found.append(name)
        assert found, (
            "the vendored scheduler no longer carries upstream's "
            f"DISTILLED_SIGMA_VALUES {DISTILLED_SIGMA_VALUES}"
        )

    def test_schedule_is_monotone_and_ends_at_zero(self):
        for values in (DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES):
            assert values[-1] == 0.0
            assert all(a > b for a, b in zip(values, values[1:]))
