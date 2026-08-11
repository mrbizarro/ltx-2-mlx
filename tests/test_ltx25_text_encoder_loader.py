"""LTX-2.5 text-encoder loader — config-keyed Gemma 3 / Gemma 4 selection.

The 2.5 text encoder is a 26 GB gated fine-tune that is not on this machine,
so these tests do the two things that do not need it: they synthesize the
*config* a 2.5 encoder declares and the *keys* its projection ships, and they
assert the loader reads both instead of assuming 2.3's.

The failure this guards against is the quiet one. If a 2.5 checkpoint's
encoder silently resolved to the Gemma 3 tower, every prompt would encode
through the wrong model. Nothing would crash. Renders would just be worse,
with no signal pointing at the cause. So ``resolve_text_tower`` is required
to raise, and there is a test that it does.
"""

from __future__ import annotations

import json
import sys

import mlx.core as mx
import pytest

from ltx_core_mlx.text_encoders.gemma.loader import (
    MODEL_TYPE_ALIASES,
    detect_projection_from_file,
    detect_text_encoder,
    detect_text_projection,
    feature_extractor_kwargs,
    read_encoder_config,
    resolve_text_tower,
)

GEMMA3_CONFIG = {
    "model_type": "gemma3",
    "hidden_size": 3840,
    "num_hidden_layers": 48,
    "num_attention_heads": 16,
    "head_dim": 256,
    "intermediate_size": 15360,
}

# Shape of the 2.5 encoder's config: multimodal packaging, text tower nested.
GEMMA4_UNIFIED_CONFIG = {
    "model_type": "gemma4_unified",
    "text_config": {
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "head_dim": 256,
        "intermediate_size": 15360,
    },
    "vision_config": {"hidden_size": 1152},
    "audio_config": {"hidden_size": 1536},
}


def write_config(tmp_path, config):
    d = tmp_path / "encoder"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    return d


class TestConfigReading:
    def test_gemma3_directory(self, tmp_path):
        spec = detect_text_encoder(write_config(tmp_path, GEMMA3_CONFIG))
        assert spec.model_type == "gemma3"
        assert spec.architecture == "gemma3"
        assert spec.generation == (2, 3)
        assert spec.hidden_size == 3840
        assert spec.num_gemma_layers == 49

    def test_gemma4_unified_directory(self, tmp_path):
        spec = detect_text_encoder(write_config(tmp_path, GEMMA4_UNIFIED_CONFIG))
        assert spec.model_type == "gemma4_unified"
        assert spec.architecture == "gemma4"
        assert spec.generation == (2, 5)

    def test_text_tower_dims_are_read_from_the_nested_config(self, tmp_path):
        """The vision/audio towers must never contribute dims — or a size."""
        spec = detect_text_encoder(write_config(tmp_path, GEMMA4_UNIFIED_CONFIG))
        assert spec.hidden_size == 3840  # not 1152 (vision) or 1536 (audio)
        assert spec.num_hidden_layers == 48

    def test_missing_config_falls_back_to_the_23_shape(self, tmp_path):
        spec = detect_text_encoder(tmp_path / "nothing-here")
        assert spec.generation == (2, 3)
        assert spec.projection_input_dim == 3840 * 49

    def test_unparseable_config_does_not_raise(self, tmp_path):
        d = tmp_path / "encoder"
        d.mkdir()
        (d / "config.json").write_text("{not json")
        assert read_encoder_config(d) == {}


class TestProjectionInputDim:
    def test_derived_never_hardcoded(self, tmp_path):
        """hidden_size * (num_hidden_layers + 1), the way ComfyUI derives it."""
        spec = detect_text_encoder(write_config(tmp_path, GEMMA3_CONFIG))
        assert spec.projection_input_dim == 3840 * 49 == 188160

    def test_gemma4_lands_on_the_same_number_which_is_a_coincidence(self, tmp_path):
        """Same dims as Gemma 3 12B — true today, not guaranteed tomorrow."""
        spec = detect_text_encoder(write_config(tmp_path, GEMMA4_UNIFIED_CONFIG))
        assert spec.projection_input_dim == 188160

    def test_a_different_encoder_moves_the_number(self, tmp_path):
        """The point of deriving it: a hardcoded 188160 would silently mis-stride."""
        config = dict(GEMMA3_CONFIG, hidden_size=2048, num_hidden_layers=32)
        spec = detect_text_encoder(write_config(tmp_path, config))
        assert spec.projection_input_dim == 2048 * 33


class TestResolveTextTower:
    def test_gemma3_resolves(self, tmp_path):
        spec = detect_text_encoder(write_config(tmp_path, GEMMA3_CONFIG))
        assert resolve_text_tower(spec) == "gemma3"

    def test_gemma4_resolves_to_the_vendored_tower(self, tmp_path):
        """Was: "fails loud when the runtime cannot build it".

        It can now — ``ltx_core_mlx.text_encoders.gemma.gemma4`` is vendored
        precisely because no mlx-lm release provides a correct gemma4 at the
        ``mlx`` version this project pins. Resolution no longer depends on the
        installed mlx-lm at all.
        """
        spec = detect_text_encoder(write_config(tmp_path, GEMMA4_UNIFIED_CONFIG))
        assert resolve_text_tower(spec) == "gemma4"

    def test_gemma4_never_silently_falls_back_to_gemma3(self, tmp_path):
        """The load-bearing refusal, unchanged: a wrong TE is undiagnosable.

        Falling back to Gemma 3 would not crash — renders would merely get
        worse, and no user could ever attribute it.
        """
        spec = detect_text_encoder(write_config(tmp_path, GEMMA4_UNIFIED_CONFIG))
        assert "gemma3" not in resolve_text_tower(spec)

    def test_gemma4_resolution_does_not_depend_on_mlx_lm(self, tmp_path, monkeypatch):
        """Hiding ``mlx_lm.models.gemma4`` must not change the answer."""
        monkeypatch.setitem(sys.modules, "mlx_lm.models.gemma4", None)
        spec = detect_text_encoder(write_config(tmp_path, GEMMA4_UNIFIED_CONFIG))
        assert resolve_text_tower(spec) == "gemma4"

    def test_unknown_model_type_raises(self, tmp_path):
        spec = detect_text_encoder(write_config(tmp_path, dict(GEMMA3_CONFIG, model_type="llama9")))
        with pytest.raises(NotImplementedError, match="Unsupported"):
            resolve_text_tower(spec)

    def test_every_alias_maps_to_a_buildable_family(self):
        for raw, resolved in MODEL_TYPE_ALIASES.items():
            assert resolved.startswith("gemma3") or resolved.startswith("gemma4"), raw


class TestProjectionDetection:
    """Keys and shapes taken from ComfyUI's sd_detect (commit 57ce8e1a)."""

    def test_dual_linear_with_bias(self):
        sd = {
            "text_embedding_projection.video_aggregate_embed.weight": mx.zeros((4096, 188160)),
            "text_embedding_projection.video_aggregate_embed.bias": mx.zeros((4096,)),
            "text_embedding_projection.audio_aggregate_embed.weight": mx.zeros((2048, 188160)),
            "text_embedding_projection.audio_aggregate_embed.bias": mx.zeros((2048,)),
        }
        spec = detect_text_projection(sd)
        assert spec.is_dual
        assert (spec.video_dim, spec.audio_dim) == (4096, 2048)
        assert spec.video_bias and spec.audio_bias
        assert spec.input_dim == 188160

    def test_bias_flags_follow_key_presence_not_a_default(self):
        """2.5 makes both bias flags configurable; presence is the truth."""
        sd = {
            "text_embedding_projection.video_aggregate_embed.weight": mx.zeros((4096, 188160)),
            "text_embedding_projection.audio_aggregate_embed.weight": mx.zeros((2048, 188160)),
            "text_embedding_projection.audio_aggregate_embed.bias": mx.zeros((2048,)),
        }
        spec = detect_text_projection(sd)
        assert spec.video_bias is False
        assert spec.audio_bias is True

    def test_single_linear(self):
        sd = {"text_embedding_projection.weight": mx.zeros((3840, 188160))}
        spec = detect_text_projection(sd)
        assert spec.projection_type == "single_linear"
        assert spec.video_dim == 3840
        assert spec.audio_dim is None
        assert spec.video_bias is False

    def test_legacy_aggregate_embed_name(self):
        sd = {"text_embedding_projection.aggregate_embed.weight": mx.zeros((3840, 188160))}
        assert detect_text_projection(sd).projection_type == "single_linear"

    def test_no_projection_returns_none(self):
        """A bare Gemma tower — the projection still lives in the connector."""
        assert detect_text_projection({"model.layers.0.self_attn.q_proj.weight": mx.zeros((4, 4))}) is None

    def test_prefix_is_honoured(self):
        sd = {
            "te.text_embedding_projection.video_aggregate_embed.weight": mx.zeros((4096, 188160)),
            "te.text_embedding_projection.audio_aggregate_embed.weight": mx.zeros((2048, 188160)),
        }
        assert detect_text_projection(sd, prefix="te.") is not None
        assert detect_text_projection(sd) is None

    def test_works_on_a_bare_key_iterable(self):
        """Callers that only have names (a header listing) still get a type."""
        keys = [
            "text_embedding_projection.video_aggregate_embed.weight",
            "text_embedding_projection.audio_aggregate_embed.weight",
        ]
        assert detect_text_projection(keys).is_dual


class TestSyntheticCheckpointRoundTrip:
    def test_detects_a_with_proj_text_encoder_file(self, tmp_path):
        """The 2.5 layout: projection relocated INTO the TE file."""
        path = tmp_path / "gemma4-12b-with-proj.safetensors"
        mx.save_safetensors(
            str(path),
            {
                "model.layers.0.self_attn.q_norm.weight": mx.zeros((8,)),
                "text_embedding_projection.video_aggregate_embed.weight": mx.zeros((16, 64)),
                "text_embedding_projection.video_aggregate_embed.bias": mx.zeros((16,)),
                "text_embedding_projection.audio_aggregate_embed.weight": mx.zeros((8, 64)),
                "text_embedding_projection.audio_aggregate_embed.bias": mx.zeros((8,)),
            },
        )
        spec = detect_projection_from_file(path)
        assert spec.is_dual
        assert (spec.video_dim, spec.audio_dim, spec.input_dim) == (16, 8, 64)

    def test_missing_file_is_none(self, tmp_path):
        assert detect_projection_from_file(tmp_path / "nope.safetensors") is None


class TestFeatureExtractorKwargs:
    def test_sizes_the_connector_from_config_and_weights(self, tmp_path):
        spec = detect_text_encoder(write_config(tmp_path, GEMMA4_UNIFIED_CONFIG))
        projection = detect_text_projection(
            {
                "text_embedding_projection.video_aggregate_embed.weight": mx.zeros((4096, 188160)),
                "text_embedding_projection.audio_aggregate_embed.weight": mx.zeros((2048, 188160)),
            }
        )
        kwargs = feature_extractor_kwargs(spec, projection)
        assert kwargs == {
            "caption_channels": 3840,
            "num_gemma_layers": 49,
            "video_dim": 4096,
            "audio_dim": 2048,
        }

    def test_without_a_projection_only_the_encoder_dims_are_set(self, tmp_path):
        spec = detect_text_encoder(write_config(tmp_path, GEMMA3_CONFIG))
        assert feature_extractor_kwargs(spec) == {"caption_channels": 3840, "num_gemma_layers": 49}

    def test_kwargs_reproduce_the_23_defaults(self, tmp_path):
        """A 2.3 encoder must size the connector exactly as it does today."""
        from ltx_core_mlx.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorV2

        spec = detect_text_encoder(write_config(tmp_path, GEMMA3_CONFIG))
        extractor = GemmaFeaturesExtractorV2(**feature_extractor_kwargs(spec))
        assert extractor.caption_channels == 3840
        assert extractor.num_gemma_layers == 49
        assert extractor.connector.text_embedding_projection.video_aggregate_embed.weight.shape == (
            4096,
            188160,
        )
