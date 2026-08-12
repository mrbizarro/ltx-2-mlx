"""Every reachable sigma schedule ends at 0.0 — and the graded presets are pinned.

Two jobs, and the first is a regression gate.

**1. The truncation bug.** Until 2026-08-12 a step count *sliced* the distilled
tables (``DISTILLED_SIGMAS[: stage1_steps + 1]``), which drops the terminal 0.0.
``--stage2-steps 2`` on LTX-2.5 therefore ran ``[0.85, 0.725, 0.421875]`` — an
**unfinished refine** that hands on a latent with residual noise — while
reporting itself as "2 steps". Anyone who reached for those flags to run a
render cheaper was grading a half-denoised latent and attributing it to the step
count (`notes/ltx25_perf_exp45.md` §B.8). The exhaustive test below walks every
table x every legal step count x every preset x every checkpoint generation and
asserts the terminal zero survives all of them, because a fix that covers one
call site and misses three is the same bug with a smaller blast radius.

**2. The adopted schedules.** Experiment 5 graded six arms on the LTX-2.5
distilled lane; the owner passed two on 2026-08-12. Arm **S2** (stage 2 at two
steps, -17 % wall, composition 0.9988 — the same take) became this lane's
default; arm **F6S2** (-29 %, a *different* take) became the ``fast`` preset.
Those exact sigma lists are pinned here so a later "cleanup" cannot quietly
re-space them, and the **2.3 lane is pinned unchanged** in the same breath —
experiment 5 never ran on a 2.3 checkpoint and says nothing about what it can
spare.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from ltx_pipelines_mlx import cli as cli_module
from ltx_pipelines_mlx import distilled as distilled_module
from ltx_pipelines_mlx.scheduler import (
    DISTILLED_MAX_POINTS,
    DISTILLED_PRESET_NAMES,
    DISTILLED_SIGMAS,
    DISTILLED_SIGMAS_LTX25_FAST,
    STAGE_2_SIGMAS,
    STAGE_2_SIGMAS_LTX25,
    STAGE_2_SIGMAS_LTX25_S2,
    distilled_presets_for,
    get_sigma_schedule,
    resolve_distilled_schedule,
    resolve_stage2_sigmas,
    thin_sigmas,
    validate_sigmas,
)

# The checkpoint generations a pipeline can actually be handed: 2.5, the older
# 2.3, and the empty tuple `model_version_of` returns for a checkpoint whose
# version it cannot read.
GENERATIONS = [(2, 5), (2, 3), ()]

TABLES = {
    "DISTILLED_SIGMAS": DISTILLED_SIGMAS,
    "STAGE_2_SIGMAS": STAGE_2_SIGMAS,
    "STAGE_2_SIGMAS_LTX25": STAGE_2_SIGMAS_LTX25,
    "STAGE_2_SIGMAS_LTX25_S2": STAGE_2_SIGMAS_LTX25_S2,
    "DISTILLED_SIGMAS_LTX25_FAST": DISTILLED_SIGMAS_LTX25_FAST,
}


def assert_usable(sigmas, context: str) -> None:
    """A schedule is usable only if it terminates at 0.0 and descends to it."""
    assert len(sigmas) >= 2, f"{context}: {sigmas} is not a schedule"
    assert sigmas[-1] == 0.0, f"{context}: {sigmas} does not terminate at 0.0"
    assert sigmas[0] <= 1.0, f"{context}: {sigmas} starts above the noise range"
    for a, b in zip(sigmas[:-1], sigmas[1:]):
        assert a > b, f"{context}: {sigmas} is not strictly decreasing"


# --------------------------------------------------------------------------- #
# 1. the regression gate — nothing reachable drops the terminal 0.0
# --------------------------------------------------------------------------- #
class TestEveryReachableScheduleTerminates:
    @pytest.mark.parametrize("name", sorted(TABLES))
    def test_the_tables_themselves(self, name):
        assert_usable(TABLES[name], name)

    @pytest.mark.parametrize("name", sorted(TABLES))
    def test_thinned_to_every_legal_step_count(self, name):
        table = TABLES[name]
        for steps in range(1, len(table)):
            thinned = thin_sigmas(table, steps)
            assert_usable(thinned, f"{name} thinned to {steps} steps")
            assert len(thinned) == steps + 1
            assert thinned[0] == table[0], "thinning must keep the first point"

    @pytest.mark.parametrize("version", GENERATIONS)
    def test_resolve_stage2_sigmas(self, version):
        for steps in (None, 0, 1, 2, 3):
            assert_usable(resolve_stage2_sigmas(version, steps), f"stage 2 {version} steps={steps}")

    @pytest.mark.parametrize("version", GENERATIONS)
    def test_resolve_distilled_schedule_over_presets_and_steps(self, version):
        for preset in distilled_presets_for(version):
            # A step count addresses the checkpoint's own table (the vendor
            # preset), so that is what bounds the legal range — not the
            # possibly-thinner preset the caller picked.
            base_1, base_2 = distilled_presets_for(version)["vendor"].as_lists()
            for s1 in [None, *range(1, len(base_1))]:
                for s2 in [None, *range(1, len(base_2))]:
                    got_1, got_2 = resolve_distilled_schedule(version, preset=preset, stage1_steps=s1, stage2_steps=s2)
                    context = f"{version} preset={preset} steps={s1}/{s2}"
                    assert_usable(got_1, f"stage 1 {context}")
                    assert_usable(got_2, f"stage 2 {context}")

    def test_get_sigma_schedule(self):
        for name in ("distilled", "stage_2", "stage_2_ltx25"):
            full = get_sigma_schedule(name)
            for steps in [None, *range(1, len(full))]:
                assert_usable(get_sigma_schedule(name, steps), f"{name} steps={steps}")

    def test_the_exact_call_that_used_to_be_broken(self):
        """``--stage2-steps 2`` on LTX-2.5: an unfinished refine, before."""
        _, stage2 = resolve_distilled_schedule((2, 5), preset="vendor", stage2_steps=2)
        assert stage2 == [0.85, 0.421875, 0.0]
        assert stage2 != STAGE_2_SIGMAS_LTX25[:3], "that slice is the bug: it stops at 0.421875"

    def test_an_existing_callers_stage2_steps_3_still_means_the_vendor_list(self):
        """The Phosphene panel — and this package's ic-lora / lipdub / keyframe
        defaults — pass ``stage2_steps=3`` explicitly. The adopted 2.5 default
        holds only 2 steps, so a step count that thinned *the preset* would turn
        those existing calls into a ValueError. A step count thins the
        CHECKPOINT's table instead, which keeps every one of them working."""
        for preset in DISTILLED_PRESET_NAMES:
            _, stage2 = resolve_distilled_schedule((2, 5), preset=preset, stage2_steps=3)
            assert stage2 == STAGE_2_SIGMAS_LTX25
        stage1, _ = resolve_distilled_schedule((2, 5), preset="fast", stage1_steps=8)
        assert stage1 == DISTILLED_SIGMAS

    def test_the_other_broken_call(self):
        """``--stage1-steps 5`` used to hand the upscaler a latent at sigma 0.909."""
        stage1, _ = resolve_distilled_schedule((2, 5), stage1_steps=5)
        assert stage1[-1] == 0.0
        assert stage1 != DISTILLED_SIGMAS[:6]

    def test_no_pipeline_still_slices_a_sigma_table_by_step_count(self):
        """The wart, grepped for by shape, across the whole pipelines package.

        A fix applied to one call site while three others keep truncating is the
        same bug. This walks the package instead of trusting the four edits.
        """
        package = Path(distilled_module.__file__).parent
        offenders = []
        pattern = re.compile(r"SIGMAS\[\s*:\s*stage\d_steps")
        for path in package.rglob("*.py"):
            if pattern.search(path.read_text()):
                offenders.append(path.name)
        assert offenders == [], f"truncating slice is back in: {offenders}"

    def test_the_distilled_pipeline_reaches_the_resolver(self):
        source = inspect.getsource(distilled_module.DistilledPipeline.generate_two_stage)
        assert "resolve_distilled_schedule(" in source
        # Both denoise loops must be fed the resolved lists, not a re-slice.
        assert source.count("sigmas=sigmas_1") == 1
        assert source.count("sigmas=sigmas_2") == 1


# --------------------------------------------------------------------------- #
# 2. thinning semantics + refusals
# --------------------------------------------------------------------------- #
class TestThinning:
    def test_full_step_count_is_the_identity(self):
        assert thin_sigmas(DISTILLED_SIGMAS, 8) == DISTILLED_SIGMAS
        assert thin_sigmas(STAGE_2_SIGMAS_LTX25, 3) == STAGE_2_SIGMAS_LTX25

    def test_none_and_zero_are_the_identity(self):
        assert thin_sigmas(DISTILLED_SIGMAS, None) == DISTILLED_SIGMAS
        assert thin_sigmas(DISTILLED_SIGMAS, 0) == DISTILLED_SIGMAS

    def test_one_step_is_the_endpoints(self):
        assert thin_sigmas(DISTILLED_SIGMAS, 1) == [1.0, 0.0]

    def test_interior_points_come_from_the_table(self):
        thinned = thin_sigmas(DISTILLED_SIGMAS, 4)
        assert all(s in DISTILLED_SIGMAS for s in thinned)

    def test_returns_a_copy(self):
        thin_sigmas(DISTILLED_SIGMAS, 8).append(-1.0)
        assert DISTILLED_SIGMAS[-1] == 0.0

    def test_more_steps_than_the_table_holds_is_refused(self):
        with pytest.raises(ValueError, match="cannot thin"):
            thin_sigmas(STAGE_2_SIGMAS_LTX25, 8)

    def test_the_refusal_names_the_way_out(self):
        with pytest.raises(ValueError, match="--stage2-sigmas"):
            thin_sigmas(STAGE_2_SIGMAS_LTX25, 99)

    def test_negative_is_refused(self):
        with pytest.raises(ValueError, match="must be positive"):
            thin_sigmas(DISTILLED_SIGMAS, -2)


# --------------------------------------------------------------------------- #
# 3. explicit schedules — the additive input, and what it rejects
# --------------------------------------------------------------------------- #
class TestExplicitSchedules:
    def test_a_valid_list_is_taken_verbatim(self):
        wanted = [1.0, 0.9, 0.5, 0.0]
        stage1, _ = resolve_distilled_schedule((2, 5), stage1_sigmas=wanted)
        assert stage1 == wanted

    def test_stage_2_alone_leaves_stage_1_on_the_preset(self):
        stage1, stage2 = resolve_distilled_schedule((2, 5), stage2_sigmas=[0.85, 0.0])
        assert stage1 == DISTILLED_SIGMAS
        assert stage2 == [0.85, 0.0]

    @pytest.mark.parametrize(
        "bad,reason",
        [
            ([0.85, 0.421875], "terminate at 0.0"),
            ([0.85, 0.85, 0.0], "strictly decreasing"),
            ([0.5, 0.9, 0.0], "strictly decreasing"),
            ([1.5, 0.5, 0.0], "starts above 1.0"),
            ([0.0], "at least 2 points"),
        ],
    )
    def test_rejections(self, bad, reason):
        with pytest.raises(ValueError, match=reason):
            resolve_distilled_schedule((2, 5), stage2_sigmas=bad)

    def test_longer_than_the_checkpoint_is_refused(self):
        too_long = [1.0 - i / 20 for i in range(DISTILLED_MAX_POINTS + 1)] + [0.0]
        with pytest.raises(ValueError, match="out of\n?\\s*distribution|out of distribution"):
            resolve_distilled_schedule((2, 5), stage1_sigmas=too_long)

    def test_a_list_and_a_step_count_together_is_a_contradiction(self):
        with pytest.raises(ValueError, match="not both"):
            resolve_distilled_schedule((2, 5), stage2_sigmas=[0.85, 0.0], stage2_steps=2)

    def test_the_experiment_arms_are_all_expressible(self):
        """§B.8's finding was that no arm could be expressed. Now they all can."""
        arms = {
            "A_control": ([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0], STAGE_2_SIGMAS_LTX25),
            "F6": (DISTILLED_SIGMAS_LTX25_FAST, STAGE_2_SIGMAS_LTX25),
            "F5": ([1.0, 0.909375, 0.725, 0.421875, 0.0], STAGE_2_SIGMAS_LTX25),
            "T6": ([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.0], STAGE_2_SIGMAS_LTX25),
            "S2": (DISTILLED_SIGMAS, STAGE_2_SIGMAS_LTX25_S2),
            "F6S2": (DISTILLED_SIGMAS_LTX25_FAST, STAGE_2_SIGMAS_LTX25_S2),
        }
        for name, (want_1, want_2) in arms.items():
            got_1, got_2 = resolve_distilled_schedule((2, 5), stage1_sigmas=want_1, stage2_sigmas=want_2)
            assert (got_1, got_2) == (list(want_1), list(want_2)), name

    def test_validate_sigmas_rejects_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            validate_sigmas([1.0, float("nan"), 0.0])


# --------------------------------------------------------------------------- #
# 4. the adopted presets, pinned to the graded arms
# --------------------------------------------------------------------------- #
class TestLtx25Presets:
    def test_default_is_arm_s2(self):
        stage1, stage2 = resolve_distilled_schedule((2, 5))
        assert stage1 == DISTILLED_SIGMAS, "S2 leaves stage 1 alone — that is the point of it"
        assert stage2 == [0.85, 0.421875, 0.0]

    def test_default_is_one_forward_cheaper_than_the_vendor_list(self):
        """The distilled lane costs (len(s1)-1) + (len(s2)-1) forwards. Measured."""
        d1, d2 = resolve_distilled_schedule((2, 5), preset="default")
        v1, v2 = resolve_distilled_schedule((2, 5), preset="vendor")
        assert (len(d1) - 1) + (len(d2) - 1) == 10
        assert (len(v1) - 1) + (len(v2) - 1) == 11

    def test_fast_is_arm_f6s2(self):
        stage1, stage2 = resolve_distilled_schedule((2, 5), preset="fast")
        assert stage1 == [1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0]
        assert stage2 == [0.85, 0.421875, 0.0]
        assert (len(stage1) - 1) + (len(stage2) - 1) == 7

    def test_vendor_reproduces_what_shipped_before_adoption(self):
        stage1, stage2 = resolve_distilled_schedule((2, 5), preset="vendor")
        assert stage1 == DISTILLED_SIGMAS
        assert stage2 == STAGE_2_SIGMAS_LTX25
        assert stage2[0] == 0.85, "2.5's first sigma is the vendor template's, not 2.3's"

    def test_the_default_and_the_vendor_list_differ(self):
        """Adoption changes output. Recorded here so nobody reads 0.9988 as 'same file'."""
        assert resolve_distilled_schedule((2, 5), preset="default") != resolve_distilled_schedule(
            (2, 5), preset="vendor"
        )

    def test_an_unknown_preset_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown distilled schedule preset"):
            resolve_distilled_schedule((2, 5), preset="turbo")


# --------------------------------------------------------------------------- #
# 5. the 2.3 lane is untouched
# --------------------------------------------------------------------------- #
class TestLtx23IsPinned:
    @pytest.mark.parametrize("version", [(2, 3), ()])
    def test_default_is_the_vendor_8_plus_3(self, version):
        stage1, stage2 = resolve_distilled_schedule(version)
        assert stage1 == DISTILLED_SIGMAS
        assert stage2 == STAGE_2_SIGMAS
        assert stage2[0] == 0.909375, "2.3 keeps its own stage-2 first sigma"

    @pytest.mark.parametrize("version", [(2, 3), ()])
    def test_the_thinned_presets_are_not_offered(self, version):
        assert "fast" not in distilled_presets_for(version)
        with pytest.raises(ValueError, match="not offered on older ones"):
            resolve_distilled_schedule(version, preset="fast")

    @pytest.mark.parametrize("version", [(2, 3), ()])
    def test_vendor_and_default_are_the_same_schedule_on_2_3(self, version):
        assert resolve_distilled_schedule(version, preset="vendor") == resolve_distilled_schedule(version)

    def test_stage2_resolution_for_the_dev_lanes_is_unmoved(self):
        """`resolve_stage2_sigmas` feeds the CFG two-stage lanes, not this one."""
        assert resolve_stage2_sigmas((2, 3)) == STAGE_2_SIGMAS
        assert resolve_stage2_sigmas((2, 5)) == STAGE_2_SIGMAS_LTX25
        assert resolve_stage2_sigmas(()) == STAGE_2_SIGMAS


# --------------------------------------------------------------------------- #
# 6. the CLI surface
# --------------------------------------------------------------------------- #
class TestCliWiring:
    def test_cli_preset_choices_match_the_scheduler(self):
        assert cli_module._DISTILLED_PRESET_NAMES == DISTILLED_PRESET_NAMES

    def test_every_advertised_preset_resolves_on_2_5(self):
        for name in DISTILLED_PRESET_NAMES:
            resolve_distilled_schedule((2, 5), preset=name)

    def test_parse_accepts_the_documented_form(self):
        assert cli_module._parse_sigma_list("1.0,0.975,0.909375,0.725,0.421875,0.0", "--stage1-sigmas") == [
            1.0,
            0.975,
            0.909375,
            0.725,
            0.421875,
            0.0,
        ]

    def test_parse_tolerates_spaces(self):
        assert cli_module._parse_sigma_list("0.85, 0.421875, 0.0", "--stage2-sigmas") == [0.85, 0.421875, 0.0]

    def test_parse_refuses_an_unfinished_schedule_before_anything_loads(self):
        with pytest.raises(SystemExit, match="terminate at 0.0"):
            cli_module._parse_sigma_list("0.85,0.725,0.421875", "--stage2-sigmas")

    def test_parse_refuses_garbage(self):
        with pytest.raises(SystemExit, match="comma-separated numbers"):
            cli_module._parse_sigma_list("0.85,fast,0.0", "--stage2-sigmas")

    def test_generate_and_save_forwards_the_schedule_inputs(self):
        """The kwargs survive the generate_and_save -> generate_two_stage hop.

        Built on a bare instance: no weights, no GPU, no Gemma. Only the two
        decode calls are stubbed out, so the forwarding path under test is the
        real one.
        """
        pipe = object.__new__(distilled_module.DistilledPipeline)
        pipe.low_memory = False
        seen: dict = {}

        def record(**kwargs):
            seen.update(kwargs)
            return "video", "audio"

        pipe.generate_two_stage = record
        pipe._load_decoders = lambda: None
        pipe._decode_and_save_video = lambda *a, **kw: "out.mp4"

        assert (
            pipe.generate_and_save(
                prompt="p",
                output_path="out.mp4",
                frame_rate=24,
                stage1_sigmas=[1.0, 0.5, 0.0],
                stage2_sigmas=[0.85, 0.0],
                schedule_preset="fast",
            )
            == "out.mp4"
        )
        assert seen["stage1_sigmas"] == [1.0, 0.5, 0.0]
        assert seen["stage2_sigmas"] == [0.85, 0.0]
        assert seen["schedule_preset"] == "fast"

    def test_generate_and_save_omits_them_when_unset(self):
        """The dev lanes share this method and their generate_two_stage has no
        such parameters — an unconditional forward would break every CFG run."""
        pipe = object.__new__(distilled_module.DistilledPipeline)
        pipe.low_memory = False
        seen: dict = {}

        def record(**kwargs):
            seen.update(kwargs)
            return "video", "audio"

        pipe.generate_two_stage = record
        pipe._load_decoders = lambda: None
        pipe._decode_and_save_video = lambda *a, **kw: "out.mp4"
        pipe.generate_and_save(prompt="p", output_path="out.mp4", frame_rate=24)

        assert "stage1_sigmas" not in seen
        assert "stage2_sigmas" not in seen
        assert "schedule_preset" not in seen
