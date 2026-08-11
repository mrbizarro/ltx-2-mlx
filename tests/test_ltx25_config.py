"""LTX-2.5 structural tests — architecture delta, no weights.

The 2.5 weights are gated on Hugging Face, so nothing here loads a real
checkpoint. What these tests CAN prove is the part that actually bites: that
a 2.5 config builds a *different* parameter set than a 2.3 config, in exactly
the ways the checkpoint expects, and that a 2.3 config is untouched by any of
it.

That matters because every flag in the 2.5 delta fails silently. A model
built with the wrong ``ff_bias`` loads without error, runs without error, and
produces noise — the June 2026 mosaic in a new costume. Asserting the
parameter *names* is the only cheap check that catches it before 42 GB of
weights are involved.
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.utils as mu
import pytest

from ltx_core_mlx.model.transformer.model import (
    LTXModel,
    LTXModelConfig,
    parse_model_version,
    read_checkpoint_metadata_config,
)

# Tiny dims — this is a shape/structure suite, not a numerics suite. Real dims
# would need 42 GB and prove nothing extra about the parameter set.
TINY = dict(
    num_layers=2,
    video_dim=32,
    audio_dim=16,
    video_num_heads=2,
    audio_num_heads=2,
    video_head_dim=16,
    audio_head_dim=8,
    av_cross_num_heads=2,
    av_cross_head_dim=8,
    video_patch_channels=4,
    audio_patch_channels=4,
)


def param_names(model) -> set[str]:
    return {k for k, _ in mu.tree_flatten(model.parameters())}


@pytest.fixture
def cfg_23() -> LTXModelConfig:
    return LTXModelConfig(**TINY)


@pytest.fixture
def cfg_25() -> LTXModelConfig:
    return LTXModelConfig(
        **TINY,
        ff_bias=False,
        audio_ff_bias=False,
        use_prompt_adaln_single=False,
        use_keyframes_abs_pos_embedding=True,
        model_version=(2, 5),
    )


class TestDefaultsAreTwoThree:
    """A config with nothing said about 2.5 must be exactly the old model."""

    def test_flag_defaults(self):
        d = LTXModelConfig()
        assert d.ff_bias is True
        assert d.audio_ff_bias is True
        assert d.connector_ff_bias is True
        assert d.use_prompt_adaln_single is True
        assert d.use_keyframes_abs_pos_embedding is False
        assert d.model_version == (2, 3)

    def test_empty_checkpoint_config_keeps_23(self):
        c = LTXModelConfig.from_checkpoint_config({})
        assert (c.ff_bias, c.audio_ff_bias, c.use_prompt_adaln_single) == (True, True, True)
        assert c.use_keyframes_abs_pos_embedding is False

    def test_a_23_checkpoint_config_is_unchanged_by_the_new_keys(self):
        """The shipped 2.3 embedded_config has none of the 2.5 keys."""
        c = LTXModelConfig.from_checkpoint_config(
            {"transformer": {"num_layers": 48, "av_ca_timestep_scale_multiplier": 1000.0}}
        )
        assert c.num_layers == 48
        assert c.av_ca_timestep_scale_multiplier == 1000.0
        assert c.ff_bias is True
        assert c.use_prompt_adaln_single is True


class TestFlagsComeFromTheCheckpoint:
    def test_25_transformer_config_flips_every_flag(self):
        c = LTXModelConfig.from_checkpoint_config(
            {
                "model_version": "2.5",
                "transformer": {
                    "ff_bias": False,
                    "audio_ff_bias": False,
                    "connector_ff_bias": False,
                    "use_prompt_adaln_single": False,
                    "use_keyframes_abs_pos_embedding": True,
                },
            }
        )
        assert c.ff_bias is False
        assert c.audio_ff_bias is False
        assert c.connector_ff_bias is False
        assert c.use_prompt_adaln_single is False
        assert c.use_keyframes_abs_pos_embedding is True
        assert c.model_version == (2, 5)

    def test_model_version_lives_at_the_top_level_not_in_transformer(self):
        """Reading it from the wrong nesting level is the easy mistake."""
        c = LTXModelConfig.from_checkpoint_config({"model_version": "2.5", "transformer": {}})
        assert c.model_version == (2, 5)


class TestParseModelVersion:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2.5", (2, 5)),
            ("2.3", (2, 3)),
            ("2.4-rc2", (2, 4)),  # hyphenated pre-release maps onto its generation
            ("2.3.rc1", (2, 3)),  # dot-separated pre-release, same rule
            ("2.5.1", (2, 5, 1)),
            ("", ()),
            ("nonsense", ()),
        ],
    )
    def test_parse(self, raw, expected):
        assert parse_model_version(raw) == expected

    def test_unversioned_sorts_below_every_real_version(self):
        """The empty tuple is what makes an unknown checkpoint safe."""
        assert () < (2, 3) < (2, 4) < (2, 5)
        assert parse_model_version("garbage") < (2, 5)

    def test_ancestral_gate_matches_upstream(self):
        """Upstream: ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)."""
        since = (2, 5)
        assert not parse_model_version("2.3") >= since
        assert not parse_model_version("2.4") >= since
        assert parse_model_version("2.5") >= since
        assert parse_model_version("2.6") >= since
        assert not parse_model_version("") >= since


class TestFeedForwardBias:
    def test_25_model_has_no_ffn_biases(self, cfg_25):
        names = param_names(LTXModel(cfg_25))
        ff_biases = {n for n in names if n.endswith(".bias") and (".ff.proj_" in n or ".audio_ff.proj_" in n)}
        assert ff_biases == set(), f"2.5 must ship no FFN biases, found {sorted(ff_biases)}"

    def test_23_model_keeps_ffn_biases(self, cfg_23):
        names = param_names(LTXModel(cfg_23))
        assert "transformer_blocks.0.ff.proj_in.bias" in names
        assert "transformer_blocks.0.ff.proj_out.bias" in names
        assert "transformer_blocks.0.audio_ff.proj_in.bias" in names

    def test_ffn_weights_are_identical_either_way(self, cfg_23, cfg_25):
        """Only biases move. A shape change here would mean a real arch change."""
        n23 = {n for n in param_names(LTXModel(cfg_23)) if ".ff.proj_" in n and n.endswith(".weight")}
        n25 = {n for n in param_names(LTXModel(cfg_25)) if ".ff.proj_" in n and n.endswith(".weight")}
        assert n23 == n25

    def test_the_two_ff_flags_are_independent(self):
        """Upstream keeps them separate; a future checkpoint may differ."""
        c = LTXModelConfig(**TINY, ff_bias=False, audio_ff_bias=True)
        names = param_names(LTXModel(c))
        assert "transformer_blocks.0.ff.proj_in.bias" not in names
        assert "transformer_blocks.0.audio_ff.proj_in.bias" in names


class TestPromptAdaLNSingle:
    def test_25_drops_both_prompt_adaln_modules(self, cfg_25):
        names = param_names(LTXModel(cfg_25))
        assert not [n for n in names if n.startswith("prompt_adaln_single.")]
        assert not [n for n in names if n.startswith("audio_prompt_adaln_single.")]

    def test_23_keeps_them(self, cfg_23):
        names = param_names(LTXModel(cfg_23))
        assert "prompt_adaln_single.linear.weight" in names
        assert "audio_prompt_adaln_single.linear.weight" in names

    def test_the_static_per_block_tables_survive_either_way(self, cfg_23, cfg_25):
        """K/V modulation still exists on 2.5 — it just loses its timestep."""
        for cfg in (cfg_23, cfg_25):
            names = param_names(LTXModel(cfg))
            assert "transformer_blocks.0.prompt_scale_shift_table" in names
            assert "transformer_blocks.0.audio_prompt_scale_shift_table" in names

    def test_kv_modulation_without_the_mlp_is_the_bare_table(self, cfg_25):
        """The whole point: with no prompt AdaLN, K/V are timestep-independent."""
        from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

        table = mx.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        shift, scale = BasicAVTransformerBlock._prompt_kv_modulation(None, table, 3)
        assert mx.allclose(shift[0, 0], table[0])
        assert mx.allclose(scale[0, 0], table[1])

    def test_kv_modulation_with_the_mlp_adds_on_top(self):
        from ltx_core_mlx.model.transformer.transformer import BasicAVTransformerBlock

        table = mx.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        params = mx.ones((1, 6))  # (B, 2*dim)
        shift, scale = BasicAVTransformerBlock._prompt_kv_modulation(params, table, 3)
        assert mx.allclose(shift[0, 0], table[0] + 1.0)
        assert mx.allclose(scale[0, 0], table[1] + 1.0)


class TestKeyframeAbsPosEmbedding:
    def test_present_only_when_declared(self, cfg_23, cfg_25):
        assert "keyframes_abs_pos_embedding" not in param_names(LTXModel(cfg_23))
        assert "keyframes_abs_pos_embedding" in param_names(LTXModel(cfg_25))

    def test_shape_is_one_by_video_dim(self, cfg_25):
        m = LTXModel(cfg_25)
        assert m.keyframes_abs_pos_embedding.shape == (1, TINY["video_dim"])


class TestParameterSetDelta:
    """The delta between generations must be exactly the four flags."""

    def test_only_expected_keys_differ(self, cfg_23, cfg_25):
        n23 = param_names(LTXModel(cfg_23))
        n25 = param_names(LTXModel(cfg_25))

        only_23 = n23 - n25
        only_25 = n25 - n23

        assert only_25 == {"keyframes_abs_pos_embedding"}

        for name in only_23:
            assert name.startswith(("prompt_adaln_single.", "audio_prompt_adaln_single.")) or (
                name.endswith(".bias") and (".ff.proj_" in name or ".audio_ff.proj_" in name)
            ), f"unexpected key dropped by the 2.5 config: {name}"

    def test_the_shared_core_is_large_and_identical(self, cfg_23, cfg_25):
        """Same 22B architecture — the overlap should dominate."""
        n23 = param_names(LTXModel(cfg_23))
        n25 = param_names(LTXModel(cfg_25))
        shared = n23 & n25
        assert len(shared) > len(n23 ^ n25)


class TestCheckpointHeaderConfig:
    """2.5 carries its architecture in the safetensors header, not a JSON."""

    def test_reads_config_and_version_from_metadata(self, tmp_path):
        path = tmp_path / "transformer-2.5.safetensors"
        config = {
            "model_version": "2.5",
            "transformer": {"ff_bias": False, "use_prompt_adaln_single": False},
        }
        mx.save_safetensors(
            str(path),
            {"patchify_proj.weight": mx.zeros((4, 4))},
            metadata={"config": json.dumps(config), "model_version": "2.5"},
        )

        read = read_checkpoint_metadata_config(path)
        assert read["model_version"] == "2.5"
        assert read["transformer"]["ff_bias"] is False

        cfg = LTXModelConfig.from_checkpoint_file(path)
        assert cfg is not None
        assert cfg.ff_bias is False
        assert cfg.use_prompt_adaln_single is False
        assert cfg.model_version == (2, 5)

    def test_a_headerless_file_returns_none_so_callers_fall_through(self, tmp_path):
        """None, not defaults — the caller must reach config.json instead."""
        path = tmp_path / "transformer-2.3.safetensors"
        mx.save_safetensors(str(path), {"patchify_proj.weight": mx.zeros((4, 4))})
        assert read_checkpoint_metadata_config(path) == {}
        assert LTXModelConfig.from_checkpoint_file(path) is None

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert read_checkpoint_metadata_config(tmp_path / "nope.safetensors") == {}

    def test_unparseable_config_json_does_not_raise(self, tmp_path):
        path = tmp_path / "bad.safetensors"
        mx.save_safetensors(
            str(path),
            {"w": mx.zeros((2, 2))},
            metadata={"config": "{not json", "model_version": "2.5"},
        )
        read = read_checkpoint_metadata_config(path)
        assert read.get("model_version") == "2.5"
