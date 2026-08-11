"""Structural + numeric tests for the LTX-2.5 duration head.

The real component is 4 MB, gated, and not on disk. Everything below builds
the module at tiny dims with random weights, writes a synthetic safetensors
file using the key names taken from the merged ComfyUI implementation
(commit 57ce8e1a, ``comfy/ldm/lightricks/duration_head.py``), and loads it
back. That proves the two things a weightless test *can* prove: the key
names our loader expects are the key names the checkpoint ships, and the
forward math matches the reference.

``seconds_to_num_frames`` is tested exhaustively because it is pure integer
arithmetic on the VAE's ``8k + 1`` causal grid — an off-by-one there produces
a frame count the pipeline rejects at the very end of a render.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.model.duration_head import (
    DurationHead,
    load_duration_head,
    normalize_state_dict,
    seconds_to_num_frames,
)

TINY = dict(
    video_cross_attention_dim=32,
    audio_cross_attention_dim=16,
    pooler_hidden_dim=8,
    num_queries=1,
    num_pooler_heads=2,
    mlp_hidden=8,
)


def random_state_dict(prefix: str = "", *, dims=TINY, seed: int = 7) -> dict:
    """Synthetic weights under 2.5's real key names."""
    rng = np.random.default_rng(seed)
    h = dims["pooler_hidden_dim"]
    q = dims["num_queries"]
    m = dims["mlp_hidden"]

    def arr(*shape):
        return mx.array(rng.standard_normal(shape).astype(np.float32) * 0.1)

    return {
        f"{prefix}video_input_proj.weight": arr(h, dims["video_cross_attention_dim"]),
        f"{prefix}video_input_proj.bias": arr(h),
        f"{prefix}video_modality_emb": arr(h),
        f"{prefix}audio_input_proj.weight": arr(h, dims["audio_cross_attention_dim"]),
        f"{prefix}audio_input_proj.bias": arr(h),
        f"{prefix}audio_modality_emb": arr(h),
        f"{prefix}attention_pooler.query_tokens": arr(q, h),
        f"{prefix}attention_pooler.cross_attn.in_proj_weight": arr(3 * h, h),
        f"{prefix}attention_pooler.cross_attn.in_proj_bias": arr(3 * h),
        f"{prefix}attention_pooler.cross_attn.out_proj.weight": arr(h, h),
        f"{prefix}attention_pooler.cross_attn.out_proj.bias": arr(h),
        f"{prefix}mlp_hidden.weight": arr(m, h * q),
        f"{prefix}mlp_hidden.bias": arr(m),
        f"{prefix}mlp_out.weight": arr(1, m),
        f"{prefix}mlp_out.bias": arr(1),
    }


class TestKeyNames:
    """Names come from the checkpoint, not from our naming taste."""

    def test_module_exposes_exactly_the_checkpoint_keys(self):
        import mlx.utils as mu

        head = DurationHead(**TINY)
        ours = {k for k, _ in mu.tree_flatten(head.parameters())}
        theirs = set(random_state_dict().keys())
        assert ours == theirs, f"missing {sorted(theirs - ours)}, extra {sorted(ours - theirs)}"

    def test_packed_qkv_layout_is_preserved(self):
        """Upstream stores torch's single (3H, H) in_proj_weight. Splitting it
        into three Linears would read the same bytes and break every key."""
        head = DurationHead(**TINY)
        h = TINY["pooler_hidden_dim"]
        assert head.attention_pooler.cross_attn.in_proj_weight.shape == (3 * h, h)
        assert head.attention_pooler.cross_attn.in_proj_bias.shape == (3 * h,)


class TestNormalizeStateDict:
    @pytest.mark.parametrize("prefix", ["", "duration_head.", "model.diffusion_model.duration_head."])
    def test_strips_every_shipping_prefix(self, prefix):
        sd = normalize_state_dict(random_state_dict(prefix))
        assert "video_input_proj.weight" in sd
        assert not any(k.startswith("duration_head") for k in sd)

    def test_monolith_prefix_wins_over_the_bare_one(self):
        """Both prefixes present: the more specific one must be chosen."""
        sd = {
            "model.diffusion_model.duration_head.mlp_out.bias": mx.zeros((1,)),
            "duration_head.mlp_out.bias": mx.ones((1,)),
        }
        out = normalize_state_dict(sd)
        assert len(out) == 1
        assert float(out["mlp_out.bias"][0]) == 0.0


class TestRoundTrip:
    @pytest.mark.parametrize("prefix", ["", "duration_head.", "model.diffusion_model.duration_head."])
    def test_loads_a_synthetic_checkpoint(self, tmp_path, prefix):
        path = tmp_path / "duration-head.safetensors"
        sd = random_state_dict(prefix)
        mx.save_safetensors(str(path), sd)

        head = load_duration_head(path)
        video = mx.zeros((2, 5, TINY["video_cross_attention_dim"]))
        out = head(video_tokens=video)
        assert out.shape == (2,)

    def test_dims_are_inferred_from_the_file(self, tmp_path):
        """A checkpoint with a different pooler width must load unchanged."""
        dims = dict(TINY, pooler_hidden_dim=12, num_pooler_heads=2, mlp_hidden=6, video_cross_attention_dim=20)
        path = tmp_path / "odd.safetensors"
        mx.save_safetensors(str(path), random_state_dict(dims=dims))

        head = load_duration_head(path)
        assert head.video_input_proj.weight.shape == (12, 20)
        assert head.mlp_out.weight.shape == (1, 6)


class TestForward:
    def test_output_is_positive_seconds(self):
        """The final exp() is what makes a negative duration unrepresentable."""
        head = DurationHead(**TINY)
        head.load_weights(list(normalize_state_dict(random_state_dict()).items()))
        out = np.array(head(video_tokens=mx.array(np.random.default_rng(3).standard_normal((4, 6, 32)).astype(np.float32))))
        assert out.shape == (4,)
        assert np.all(out > 0)

    def test_accepts_either_modality_or_both(self):
        head = DurationHead(**TINY)
        head.load_weights(list(normalize_state_dict(random_state_dict()).items()))
        v = mx.zeros((1, 4, 32))
        a = mx.zeros((1, 3, 16))
        assert head(video_tokens=v).shape == (1,)
        assert head(audio_tokens=a).shape == (1,)
        assert head(video_tokens=v, audio_tokens=a).shape == (1,)

    def test_requires_at_least_one_modality(self):
        head = DurationHead(**TINY)
        with pytest.raises(ValueError, match="at least one"):
            head()

    def test_matches_a_numpy_transcription_of_the_reference(self):
        """Reference: ComfyUI duration_head.py forward(), transcribed longhand."""
        sd = normalize_state_dict(random_state_dict(seed=11))
        head = DurationHead(**TINY)
        head.load_weights(list(sd.items()))

        rng = np.random.default_rng(19)
        video = rng.standard_normal((2, 5, TINY["video_cross_attention_dim"])).astype(np.float32)

        got = np.array(head(video_tokens=mx.array(video))).astype(np.float64)

        # --- reference, NumPy ---
        n = {k: np.array(v).astype(np.float64) for k, v in sd.items()}
        h = TINY["pooler_hidden_dim"]
        heads = TINY["num_pooler_heads"]
        hd = h // heads

        tokens = video.astype(np.float64) @ n["video_input_proj.weight"].T + n["video_input_proj.bias"]
        tokens = tokens + n["video_modality_emb"]

        queries = np.broadcast_to(n["attention_pooler.query_tokens"], (tokens.shape[0], TINY["num_queries"], h))
        w, b = n["attention_pooler.cross_attn.in_proj_weight"], n["attention_pooler.cross_attn.in_proj_bias"]
        q = queries @ w[:h].T + b[:h]
        k = tokens @ w[h : 2 * h].T + b[h : 2 * h]
        v = tokens @ w[2 * h :].T + b[2 * h :]

        def split(x):
            B, T, _ = x.shape
            return x.reshape(B, T, heads, hd).transpose(0, 2, 1, 3)

        q, k, v = split(q), split(k), split(v)
        logits = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        attn = weights @ v
        B = attn.shape[0]
        attn = attn.transpose(0, 2, 1, 3).reshape(B, TINY["num_queries"], h)
        pooled = attn @ n["attention_pooler.cross_attn.out_proj.weight"].T + n["attention_pooler.cross_attn.out_proj.bias"]

        pooled = pooled.reshape(B, -1)
        hidden = pooled @ n["mlp_hidden.weight"].T + n["mlp_hidden.bias"]
        # GELU tanh approximation, as in the reference
        hidden = 0.5 * hidden * (1 + np.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * hidden**3)))
        want = np.exp((hidden @ n["mlp_out.weight"].T + n["mlp_out.bias"]).squeeze(-1))

        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


class TestSecondsToNumFrames:
    """The 8k+1 causal grid. num_frames % 8 == 1 is a hard pipeline constraint."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            # It FLOORS onto the grid, it does not round to nearest. 5 s at
            # 24 fps is 120 raw frames, and the grid point at or below 120 is
            # 113 — not the 121 the panel's "5 s" preset uses. Worth pinning:
            # a duration-head suggestion will look one grid step short of the
            # hand-picked presets, and that is correct behaviour, not a bug.
            (5.0, 113),
            # 5.05 s -> round(121.2) = 121 raw, which IS a grid point.
            (5.05, 121),
            # At the floor the bump-up branch fires: 1 s = 24 raw frames
            # floors to 17, below min_frames=24, so it climbs to 25 rather
            # than returning a length the caller asked not to go below.
            (1.0, 25),
        ],
    )
    def test_known_values(self, seconds, expected):
        got = seconds_to_num_frames(seconds, frame_rate=24, min_seconds=1.0, max_seconds=20.0)
        assert (got - 1) % 8 == 0
        assert got == expected

    @pytest.mark.parametrize("seconds", [0.1, 0.5, 1.0, 2.5, 5.0, 7.3, 10.0, 15.0, 30.0, 120.0])
    def test_always_lands_on_the_grid(self, seconds):
        got = seconds_to_num_frames(seconds, frame_rate=24, min_seconds=1.0, max_seconds=20.0)
        assert (got - 1) % 8 == 0, f"{got} is not 8k+1"

    def test_clamps_to_the_range(self):
        low = seconds_to_num_frames(0.01, frame_rate=24, min_seconds=2.0, max_seconds=10.0)
        high = seconds_to_num_frames(999.0, frame_rate=24, min_seconds=2.0, max_seconds=10.0)
        assert low >= round(2.0 * 24) - 8
        assert high <= round(10.0 * 24)
        assert (low - 1) % 8 == 0 and (high - 1) % 8 == 0

    def test_a_floor_that_undershoots_the_minimum_bumps_up(self):
        """Otherwise a short prediction snaps below the pipeline's floor."""
        got = seconds_to_num_frames(0.2, frame_rate=24, min_seconds=0.5, max_seconds=10.0)
        assert got >= 9  # next grid point at or above round(0.5*24)=12 -> 17, floor 9
        assert (got - 1) % 8 == 0

    def test_monotone_in_seconds(self):
        prev = 0
        for s in np.arange(1.0, 20.0, 0.37):
            got = seconds_to_num_frames(float(s), frame_rate=24, min_seconds=1.0, max_seconds=20.0)
            assert got >= prev
            prev = got
