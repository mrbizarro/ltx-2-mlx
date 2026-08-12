"""Contract gate for the live preview + early abort.

Runs without the DiT, without the 22 MB tiny-decoder checkpoint and without a meaningful
GPU load, so the file protocol a panel codes against cannot drift silently. The picture
quality half is falsified by ``scripts/probe_preview_context.py`` on a real latent; this
file gates the parts a test can own: the schema, the atomic write, the sentinel semantics,
the geometry arithmetic and the estimate counters.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from ltx_pipelines_mlx.live_preview import (
    ABORT_EXIT_CODE,
    SCHEMA,
    LivePreviewAborted,
    LivePreviewMonitor,
    approximate_output_frame,
    auto_downscale,
    local_output_index,
    pool_latent,
)
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS_LTX25
from ltx_pipelines_mlx.tiny_video_vae import TinyLTXVideoDecoder
from ltx_pipelines_mlx.utils.samplers import euler_loop_estimates, res2s_loop_estimates


class _StubDecoder:
    """Stands in for the tiny decoder: right shapes, no weights, no real work."""

    latent_channels = 128

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def decode(self, latents: mx.array, num_frames: int | None = None) -> mx.array:
        self.calls.append((latents.shape, num_frames))
        b, _c, t, h, w = latents.shape
        frames = t * 8 - 7 if num_frames is None else num_frames
        return mx.zeros((b, 3, frames, h * 32, w * 32))


def _monitor(tmp_path, **kwargs) -> LivePreviewMonitor:
    return LivePreviewMonitor(
        tmp_path / "live",
        None,
        output=tmp_path / "clip.mp4",
        decoder=_StubDecoder(),
        **kwargs,
    )


def _status(monitor) -> dict:
    return json.loads(monitor.status_path.read_text())


# --- geometry --------------------------------------------------------------------------


def test_latent_frame_maps_to_the_pixel_frame_the_vae_would_produce():
    # LTX groups pixel frames (1, 8, 8, ...) per latent frame: the same arithmetic
    # compute_video_positions uses, max(0, i * 8 - 7).
    assert approximate_output_frame(0) == 0
    assert approximate_output_frame(1) == 1
    assert approximate_output_frame(2) == 9
    assert approximate_output_frame(6) == 41


def test_the_previewed_token_lands_on_the_last_decoded_frame():
    # T tokens decode to T*8-7 frames, so the target token's last frame is index 8*context.
    for context in (0, 1, 2, 4):
        assert local_output_index(context) == 8 * context
        assert local_output_index(context) + 1 <= (context + 1) * 8 - 7 + 1


def test_auto_downscale_leaves_the_draft_tier_alone_and_halves_the_hd_stage_two():
    assert auto_downscale(14, 24) == 1  # 768x448 single/full-res grid = 336 cells
    assert auto_downscale(9, 16) == 1  # 1024x576 stage-1 half-res grid = 144 cells
    assert auto_downscale(18, 32) == 2  # 1024x576 stage-2 grid = 576 cells -> 144
    # An axis that cannot be halved is left alone rather than silently mis-sliced.
    assert auto_downscale(15, 33) == 1


def test_pooling_averages_and_keeps_the_channel_axis():
    latents = mx.arange(2 * 4 * 4, dtype=mx.float32).reshape(1, 1, 2, 4, 4)
    pooled = pool_latent(latents, 2)
    assert pooled.shape == (1, 1, 2, 2, 2)
    assert float(pooled[0, 0, 0, 0, 0]) == pytest.approx((0 + 1 + 4 + 5) / 4)


# --- estimate counters -----------------------------------------------------------------


def test_estimate_counters_match_the_shipped_schedules():
    # 9 sigmas = 8 steps = 8 Euler estimates.
    assert euler_loop_estimates(DISTILLED_SIGMAS) == 8
    assert euler_loop_estimates(STAGE_2_SIGMAS_LTX25) == 3
    # res_2s is second-order: 2 per step, +1 terminal when the schedule ends at 0.
    assert res2s_loop_estimates([1.0, 0.5, 0.0]) == 5
    assert res2s_loop_estimates([1.0, 0.5, 0.25]) == 4


# --- file contract ---------------------------------------------------------------------


def test_status_json_carries_the_schema_and_the_plan_before_any_forward(tmp_path):
    monitor = _monitor(tmp_path)
    monitor.plan([("stage1", 8), ("stage2", 3)])
    status = _status(monitor)
    assert status["schema"] == SCHEMA
    assert status["status"] == "starting"
    assert status["aborted"] is False
    assert status["total_forwards"] == 11
    assert status["total_stages"] == 2
    assert status["output"].endswith("clip.mp4")
    assert status["abort_sentinel"].endswith("ABORT")


def test_a_published_forward_writes_a_png_and_a_stable_latest_path(tmp_path):
    monitor = _monitor(tmp_path)
    monitor.plan([("stage1", 2)])
    monitor.start_stage("stage1", latent_frames=7, latent_height=9, latent_width=16)
    tokens = mx.zeros((1, 7 * 9 * 16, 128))

    monitor.publish(tokens, 1.0)
    status = _status(monitor)
    assert status["forward"] == 1
    assert status["stage"] == "stage1"
    assert status["preview"] == "preview_01.png"
    assert status["preview_width"] == 16 * 32
    assert status["preview_height"] == 9 * 32
    assert status["latent_frame"] == 3
    assert (monitor.directory / "preview_01.png").exists()
    latest = (monitor.directory / "preview_latest.png").read_bytes()
    assert latest == (monitor.directory / "preview_01.png").read_bytes()

    monitor.publish(tokens, 0.5)
    assert _status(monitor)["preview"] == "preview_02.png"
    assert (monitor.directory / "preview_latest.png").read_bytes() == (
        monitor.directory / "preview_02.png"
    ).read_bytes()


def test_the_decoder_is_handed_context_plus_one_tokens(tmp_path):
    monitor = _monitor(tmp_path, context=2)
    monitor.plan([("stage1", 1)])
    monitor.start_stage("stage1", latent_frames=7, latent_height=9, latent_width=16)
    monitor.publish(mx.zeros((1, 7 * 9 * 16, 128)), 1.0)
    shape, num_frames = monitor.tae.calls[0]
    assert shape[2] == 3  # 2 warm-up frames + the previewed one
    assert num_frames == local_output_index(2) + 1


def test_every_n_skips_but_still_publishes_the_last_of_a_stage(tmp_path):
    monitor = _monitor(tmp_path, every=3)
    monitor.plan([("stage1", 4)])
    monitor.start_stage("stage1", latent_frames=5, latent_height=9, latent_width=16)
    tokens = mx.zeros((1, 5 * 9 * 16, 128))
    for _ in range(4):
        monitor.publish(tokens, 1.0)
    written = sorted(p.name for p in monitor.directory.glob("preview_*.png") if "latest" not in p.name)
    assert written == ["preview_03.png", "preview_04.png"]


def test_a_stale_sentinel_from_a_previous_render_is_cleared_not_obeyed(tmp_path):
    (tmp_path / "live").mkdir(parents=True)
    (tmp_path / "live" / "ABORT").write_text("")
    monitor = _monitor(tmp_path)
    assert not monitor.abort_path.exists()
    assert _status(monitor)["stale_abort_cleared"] is True
    monitor.check_abort()  # must not raise


def test_the_sentinel_aborts_once_and_is_consumed(tmp_path):
    monitor = _monitor(tmp_path)
    monitor.plan([("stage1", 2)])
    monitor.start_stage("stage1", latent_frames=5, latent_height=9, latent_width=16)
    monitor.abort_path.write_text("")

    with pytest.raises(LivePreviewAborted):
        monitor.check_abort("before step 1/2")

    assert not monitor.abort_path.exists(), "a leftover sentinel would kill the next render"
    status = _status(monitor)
    assert status["status"] == "aborted"
    assert status["aborted"] is True
    assert status["aborted_at_stage"] == "before step 1/2"
    monitor.check_abort()  # consumed: the second check is quiet


def test_abort_exit_code_is_distinct_from_success_and_from_a_traceback():
    assert ABORT_EXIT_CODE == 75
    assert ABORT_EXIT_CODE not in (0, 1)


def test_a_latent_frame_outside_the_render_is_rejected(tmp_path):
    monitor = _monitor(tmp_path, latent_frame=99)
    with pytest.raises(ValueError, match="outside this render"):
        monitor.start_stage("stage1", latent_frames=7, latent_height=9, latent_width=16)


# --- the tiny decoder's own contract ---------------------------------------------------


def test_the_tiny_decoder_expands_a_latent_by_32x_space_and_8x_time():
    decoder = TinyLTXVideoDecoder()
    assert decoder.latent_channels == 128
    assert decoder.patch_size * 2**3 == 32
    assert decoder.temporal_upscale == 8
    out = decoder.decode(mx.zeros((1, 128, 3, 2, 2)))
    mx.eval(out)
    # 3 latent tokens -> 3*8 - 7 = 17 pixel frames, 2 latent cells -> 64 px
    assert out.shape == (1, 3, 17, 64, 64)


def test_the_tiny_decoder_rejects_a_latent_of_the_wrong_width():
    with pytest.raises(ValueError, match=r"\(B, 128, T, H, W\)"):
        TinyLTXVideoDecoder().decode(mx.zeros((1, 24, 3, 2, 2)))
