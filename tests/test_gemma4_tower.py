"""Structural proof of the vendored Gemma 4 text tower — no real weights involved.

The LTX-2.5 encoder is a gated 26 GB download nobody here has seen. These tests
prove the *architecture* instead, at tiny dims with random weights, by asserting
every documented difference between Gemma 4 and Gemma 3 against a longhand
transcription of the reference formulas.

That framing is the point. A text encoder has no crash mode: get the RoPE width,
the V path or the attention scale wrong and the tower still builds, still loads,
still encodes — just wrongly, and the only symptom is that renders are worse in a
way no user could attribute. So each delta below is pinned individually.

References, which agree with each other throughout:
  * ``mlx_lm/models/gemma4_text.py`` @ main (Apple, MIT)
  * ``comfy/text_encoders/gemma4.py`` (ComfyUI) — the PyTorch reference

The reference formulas are re-derived here in NumPy, independently of the MLX
code under test, so a shared misreading cannot pass both.
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from ltx_core_mlx.text_encoders.gemma.gemma4 import (
    FULL_ATTENTION,
    GEMMA4_12B_REFERENCE,
    SLIDING_ATTENTION,
    Gemma4Attention,
    Gemma4DecoderLayer,
    Gemma4TextConfig,
    Gemma4TextTower,
    ProportionalRoPE,
    RMSNormNoScale,
    build_rope,
    sanitize_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def tiny_config(**overrides) -> Gemma4TextConfig:
    """A 12B-shaped tower shrunk until it fits in a unit test.

    Shape-faithful where shape matters: k_eq_v on, distinct sliding/global head
    dims, a real sliding/full layer pattern.
    """
    base = dict(
        hidden_size=32,
        num_hidden_layers=6,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        attention_k_eq_v=True,
        head_dim=8,
        global_head_dim=16,
        sliding_window=4,
        sliding_window_pattern=3,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        vocab_size=100,
    )
    base.update(overrides)
    return Gemma4TextConfig(**base)


def param_names(module: nn.Module) -> set[str]:
    return {name for name, _ in tree_flatten(module.parameters())}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_layer_pattern_puts_full_attention_last_in_each_window(self):
        """5 sliding : 1 full, full-attention layer last — Gemma 3 uses 6."""
        config = tiny_config(num_hidden_layers=12, sliding_window_pattern=6)
        assert config.layer_types == (
            [SLIDING_ATTENTION] * 5 + [FULL_ATTENTION]
        ) * 2

    def test_comfy_sliding_attention_list_is_translated(self):
        """ComfyUI spells the pattern as window sizes with False for full."""
        config = Gemma4TextConfig.from_dict(
            {
                "hidden_size": 32,
                "num_hidden_layers": 6,
                "sliding_attention": [1024, 1024, False],
            }
        )
        assert config.layer_types == [
            SLIDING_ATTENTION,
            SLIDING_ATTENTION,
            FULL_ATTENTION,
        ] * 2
        assert config.sliding_window == 1024

    def test_nested_text_config_wins_over_decoy_top_level_dims(self):
        """``gemma4_unified`` nests the text tower beside vision/audio decoys.

        Reading ``hidden_size`` off the top level sizes every projection wrong
        and raises nothing at load time.
        """
        config = Gemma4TextConfig.from_dict(
            {
                "model_type": "gemma4_unified",
                "hidden_size": 9999,
                "num_hidden_layers": 1,
                "vision_config": {"hidden_size": 1152},
                "audio_config": {"hidden_size": 640},
                "text_config": {"hidden_size": 3840, "num_hidden_layers": 48},
            }
        )
        assert config.hidden_size == 3840
        assert config.num_hidden_layers == 48

    def test_twelve_b_reference_reproduces_the_projection_width(self):
        """``hidden_size * (num_layers + 1)`` must land on LTX's 188160."""
        config = Gemma4TextConfig.from_dict(dict(GEMMA4_12B_REFERENCE))
        assert config.hidden_size * (config.num_hidden_layers + 1) == 188160

    def test_mismatched_layer_types_length_is_rejected(self):
        with pytest.raises(ValueError, match="layer_types"):
            Gemma4TextConfig(num_hidden_layers=4, layer_types=[SLIDING_ATTENTION] * 3)

    def test_kv_sharing_boundary_and_head_selection(self):
        config = tiny_config(num_hidden_layers=6, num_kv_shared_layers=2)
        assert config.first_kv_shared_layer == 4
        # Full-attention layers are wider per head and use the global KV count.
        assert config.head_dim_for(FULL_ATTENTION) == 16
        assert config.head_dim_for(SLIDING_ATTENTION) == 8
        assert config.uses_k_eq_v(FULL_ATTENTION) is True
        assert config.uses_k_eq_v(SLIDING_ATTENTION) is False
        assert config.kv_heads_for(FULL_ATTENTION) == 1
        assert config.kv_heads_for(SLIDING_ATTENTION) == 2

    def test_no_sharing_means_every_layer_owns_its_kv(self):
        config = tiny_config(num_kv_shared_layers=0)
        assert config.first_kv_shared_layer == config.num_hidden_layers


# ---------------------------------------------------------------------------
# RoPE — the largest silent-failure surface
# ---------------------------------------------------------------------------


def reference_partial_rope(
    x: np.ndarray,
    dims: int,
    partial_rotary_factor: float,
    theta: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Longhand transcription of ComfyUI's partial RoPE. Independent of the MLX code.

    From ``Gemma4Transformer.__init__`` + ``_apply_rotary_pos_emb``::

        rope_angles = int(partial_rotary_factor * dims // 2)
        nope        = dims // 2 - rope_angles
        inv_freq    = 1 / theta ** (arange(0, 2*rope_angles, 2) / dims)
        inv_freq    = concat([inv_freq, zeros(nope)])
        emb         = concat([freqs, freqs]); cos, sin = emb.cos(), emb.sin()
        out[..., :h] = x[..., :h]*cos[..., :h] - x[..., h:]*sin[..., :h]
        out[..., h:] = x[..., h:]*cos[..., h:] + x[..., :h]*sin[..., h:]

    The exponent divides by the FULL ``dims``, not by the rotated width. That is
    what "proportional" means and it is the detail a from-scratch rewrite gets
    wrong while still producing a plausible-looking rotation.
    """
    rope_angles = int(partial_rotary_factor * dims // 2)
    nope = dims // 2 - rope_angles
    inv_freq = 1.0 / (theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float64) / dims))
    if nope > 0:
        inv_freq = np.concatenate([inv_freq, np.zeros(nope, dtype=np.float64)])

    freqs = positions[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)

    half = dims // 2
    out = np.empty_like(x, dtype=np.float64)
    out[..., :half] = x[..., :half] * cos[..., :half] - x[..., half:] * sin[..., :half]
    out[..., half:] = x[..., half:] * cos[..., half:] + x[..., :half] * sin[..., half:]
    return out


class TestProportionalRoPE:
    @pytest.mark.parametrize(
        ("dims", "factor", "theta"),
        [(16, 0.25, 1_000_000.0), (512, 0.25, 1_000_000.0), (32, 0.5, 10_000.0)],
    )
    def test_matches_the_reference_formula_elementwise(self, dims, factor, theta):
        """The whole rotation, against the longhand transcription."""
        seq = 7
        rope = ProportionalRoPE(dims=dims, rotated_dims=int(dims * factor), base=theta)

        x_np = np.random.default_rng(0).standard_normal((1, 1, seq, dims))
        got = np.asarray(rope(mx.array(x_np, dtype=mx.float32)).astype(mx.float32))
        want = reference_partial_rope(
            x_np, dims, factor, theta, np.arange(seq, dtype=np.float64)
        )
        assert np.allclose(got, want, atol=1e-4, rtol=1e-4)

    def test_unrotated_tail_is_exactly_identity(self):
        """The NoPE dims must not move at all — not 'almost'.

        Both halves are padded, so the untouched slice is
        ``[rotated//2 : dims//2]`` mirrored into the second half.
        """
        dims, rotated = 16, 4
        rope = ProportionalRoPE(dims=dims, rotated_dims=rotated, base=1_000_000.0)
        x = mx.random.normal((1, 1, 5, dims))
        y = rope(x, offset=3)

        half, r_half = dims // 2, rotated // 2
        for lo, hi in ((r_half, half), (half + r_half, dims)):
            assert mx.array_equal(y[..., lo:hi], x[..., lo:hi]), (
                f"dims [{lo}:{hi}] must be untouched by partial RoPE"
            )

    def test_rotated_head_actually_moves(self):
        """Guards the mirror image: a no-op RoPE would also pass the test above."""
        rope = ProportionalRoPE(dims=16, rotated_dims=4, base=1_000_000.0)
        x = mx.random.normal((1, 1, 5, 16))
        y = rope(x, offset=3)
        assert not mx.allclose(y[..., :2], x[..., :2])

    def test_full_factor_equals_plain_rope(self):
        """``partial_rotary_factor=1`` must collapse onto ordinary RoPE."""
        dims = 16
        proportional = ProportionalRoPE(dims=dims, rotated_dims=dims, base=10_000.0)
        plain = nn.RoPE(dims, traditional=False, base=10_000.0)
        x = mx.random.normal((1, 1, 6, dims))
        assert mx.allclose(proportional(x, offset=2), plain(x, offset=2), atol=1e-5)

    def test_rotated_dims_must_fit(self):
        with pytest.raises(ValueError, match="rotated_dims"):
            ProportionalRoPE(dims=8, rotated_dims=16)


class TestBuildRope:
    def test_layer_types_get_different_rope_families(self):
        """Full attention is partial+1e6; sliding is full rotary+1e4."""
        config = tiny_config()
        full = build_rope(config, FULL_ATTENTION)
        sliding = build_rope(config, SLIDING_ATTENTION)

        assert isinstance(full, ProportionalRoPE)
        assert full.dims == config.global_head_dim
        assert full.rotated_dims == int(config.global_head_dim * 0.25)
        assert isinstance(sliding, nn.RoPE)

    def test_unknown_rope_type_raises_rather_than_defaulting(self):
        config = tiny_config()
        config.rope_parameters[FULL_ATTENTION] = {
            "rope_type": "hypercube",
            "rope_theta": 10.0,
            "partial_rotary_factor": 1.0,
        }
        with pytest.raises(ValueError, match="Unsupported RoPE type"):
            build_rope(config, FULL_ATTENTION)


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class TestNorms:
    def test_value_norm_carries_no_learnable_gain(self):
        """Gemma 4 norms V with a scale-FREE RMSNorm.

        Giving it a weight adds a tensor the checkpoint does not contain, which
        loads as random noise instead of failing.
        """
        assert param_names(RMSNormNoScale(eps=1e-6)) == set()

    def test_value_norm_matches_rms_without_gain(self):
        x = mx.random.normal((2, 5))
        got = RMSNormNoScale(eps=1e-6)(x)
        want = np.asarray(x) / np.sqrt((np.asarray(x) ** 2).mean(-1, keepdims=True) + 1e-6)
        assert np.allclose(np.asarray(got), want, atol=1e-5)

    def test_block_has_four_norms_and_a_layer_scalar(self):
        """Gemma 3 has the same four; ``layer_scalar`` is the Gemma 4 addition."""
        names = param_names(Gemma4DecoderLayer(tiny_config(), layer_idx=0))
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            assert f"{norm}.weight" in names
        assert "layer_scalar" in names


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class TestAttention:
    def test_k_eq_v_layers_have_no_value_projection(self):
        """No ``v_proj`` tensor exists in the checkpoint for these layers."""
        config = tiny_config()
        full = Gemma4Attention(config, layer_idx=config.layer_types.index(FULL_ATTENTION))
        sliding = Gemma4Attention(config, layer_idx=config.layer_types.index(SLIDING_ATTENTION))

        assert "v_proj.weight" not in param_names(full)
        assert "v_proj.weight" in param_names(sliding)
        assert full.use_k_eq_v is True and sliding.use_k_eq_v is False

    def test_attention_scale_is_one(self):
        """Gemma 4 folds the temperature into q_norm; 1/sqrt(d) on top is wrong."""
        config = tiny_config()
        assert Gemma4Attention(config, layer_idx=0).scale == 1.0

    def test_head_dim_follows_layer_type(self):
        config = tiny_config()
        full_idx = config.layer_types.index(FULL_ATTENTION)
        sliding_idx = config.layer_types.index(SLIDING_ATTENTION)
        assert Gemma4Attention(config, full_idx).head_dim == config.global_head_dim
        assert Gemma4Attention(config, sliding_idx).head_dim == config.head_dim

    def test_value_is_the_raw_k_projection_not_the_normed_or_roped_one(self):
        """The single most camouflaged detail in the whole tower.

        On a ``k_eq_v`` layer, ``V`` is ``k_proj(x)`` taken **before** ``k_norm``
        and **before** RoPE, then passed through the scale-free RMSNorm. Norming
        it with ``k_norm`` or rotating it produces a tower that runs and encodes
        subtly wrongly — no shape ever disagrees.

        Proved by driving ``o_proj`` to identity and reading V back out of the
        attention output at a position that can only attend to itself.
        """
        config = tiny_config(
            num_attention_heads=1,
            num_global_key_value_heads=1,
            global_head_dim=8,
            hidden_size=8,
        )
        layer_idx = config.layer_types.index(FULL_ATTENTION)
        attn = Gemma4Attention(config, layer_idx)

        # Identity o_proj so the output IS the attended value.
        attn.o_proj.weight = mx.eye(8)
        # Non-trivial k_norm gain: if V were k_norm'd, this would show up.
        attn.k_norm.weight = mx.full((8,), 3.0)
        mx.eval(attn.parameters())

        x = mx.random.normal((1, 1, 8))  # single token -> attends only to itself
        out, _ = attn(x)

        raw_k = attn.k_proj(x).reshape(1, 1, 1, 8)
        expected = attn.v_norm(raw_k).transpose(0, 2, 1, 3).reshape(1, 1, 8)
        assert mx.allclose(out, expected, atol=1e-5), "V must be the RAW k projection"

        # And it must NOT be the k_norm'd + RoPE'd K that attention uses as keys.
        wrong = attn.rope(attn.k_norm(raw_k).transpose(0, 2, 1, 3)).reshape(1, 1, 8)
        assert not mx.allclose(out, wrong, atol=1e-3)

    def test_kv_shared_layer_owns_no_kv_parameters(self):
        config = tiny_config(num_hidden_layers=6, num_kv_shared_layers=2)
        shared = Gemma4Attention(config, layer_idx=5)
        names = param_names(shared)
        assert shared.has_kv is False
        for absent in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
            assert absent not in names
        assert "q_proj.weight" in names and "o_proj.weight" in names

    def test_kv_shared_layer_without_shared_kv_refuses(self):
        """Loud, not silent: there is no correct K/V to invent here."""
        config = tiny_config(num_hidden_layers=6, num_kv_shared_layers=2)
        shared = Gemma4Attention(config, layer_idx=5)
        with pytest.raises(ValueError, match="KV-shared"):
            shared(mx.zeros((1, 3, config.hidden_size)))


# ---------------------------------------------------------------------------
# Decoder layer / tower
# ---------------------------------------------------------------------------


class TestLayerAndTower:
    def test_layer_scalar_scales_the_block_output(self):
        config = tiny_config()
        layer = Gemma4DecoderLayer(config, layer_idx=0)
        mx.eval(layer.parameters())
        x = mx.random.normal((1, 4, config.hidden_size))

        baseline, _ = layer(x)
        layer.layer_scalar = mx.array([2.0])
        doubled, _ = layer(x)
        assert mx.allclose(doubled, 2.0 * baseline, atol=1e-5)

    def test_all_hidden_states_returns_layers_plus_one(self):
        """The LTX projection is sized ``hidden * (num_layers + 1)``."""
        config = tiny_config()
        tower = Gemma4TextTower(config)
        states = tower.all_hidden_states(mx.array([[1, 2, 3, 4]]), eval_every=0)
        assert len(states) == config.num_hidden_layers + 1
        assert all(s.shape == (1, 4, config.hidden_size) for s in states)

    def test_first_hidden_state_is_the_scaled_embedding(self):
        """Scaled by sqrt(hidden), rounded through bf16 as both references do."""
        config = tiny_config()
        tower = Gemma4TextTower(config)
        ids = mx.array([[1, 2, 3]])
        states = tower.all_hidden_states(ids, eval_every=0)

        scale = mx.array(config.hidden_size**0.5, dtype=mx.bfloat16).astype(mx.float32)
        assert mx.allclose(states[0], tower.embed_tokens(ids) * scale, atol=1e-5)

    def test_padding_mask_changes_the_encoding(self):
        config = tiny_config()
        tower = Gemma4TextTower(config)
        mx.eval(tower.parameters())
        ids = mx.array([[5, 6, 7, 8]])
        unmasked = tower.all_hidden_states(ids, eval_every=0)[-1]
        masked = tower.all_hidden_states(ids, mx.array([[0, 0, 1, 1]]), eval_every=0)[-1]
        assert not mx.allclose(unmasked, masked, atol=1e-4)

    def test_kv_sharing_routes_to_the_last_non_shared_layer_of_the_same_type(self):
        """Sliding layers borrow from sliding, full from full — never crossed."""
        config = tiny_config(num_hidden_layers=9, sliding_window_pattern=3, num_kv_shared_layers=3)
        tower = Gemma4TextTower(config)
        # types: S S F S S F | S S F  (boundary at 6)
        assert config.layer_types[:6] == [
            SLIDING_ATTENTION, SLIDING_ATTENTION, FULL_ATTENTION,
            SLIDING_ATTENTION, SLIDING_ATTENTION, FULL_ATTENTION,
        ]
        assert tower.kv_source[:6] == [0, 1, 2, 3, 4, 5]  # own their K/V
        assert tower.kv_source[6] == 4  # last non-shared sliding
        assert tower.kv_source[7] == 4
        assert tower.kv_source[8] == 5  # last non-shared full

    def test_kv_shared_tower_runs_end_to_end(self):
        config = tiny_config(num_hidden_layers=9, sliding_window_pattern=3, num_kv_shared_layers=3)
        tower = Gemma4TextTower(config)
        states = tower.all_hidden_states(mx.array([[1, 2, 3, 4, 5]]), eval_every=0)
        mx.eval(states[-1])
        assert len(states) == 10
        assert bool(mx.all(mx.isfinite(states[-1])))

    def test_sliding_window_restricts_only_the_sliding_layers(self):
        """Beyond the window, a sliding layer must not see the oldest tokens."""
        config = tiny_config(sliding_window=2)
        tower = Gemma4TextTower(config)
        masks = tower._build_masks(5, None, mx.float32)

        sliding = np.asarray(masks[SLIDING_ATTENTION])[0, 0]
        full = np.asarray(masks[FULL_ATTENTION])[0, 0]
        # Query 4 may see keys 3 and 4 only; the full-attention layer sees all.
        assert sliding[4, 2] < -1e8 and sliding[4, 3] == 0.0
        assert full[4, 0] == 0.0

    def test_masks_never_saturate_to_negative_infinity(self):
        """Additive masks must stay finite or a fully-padded row softmaxes to NaN."""
        tower = Gemma4TextTower(tiny_config(sliding_window=2))
        masks = tower._build_masks(6, mx.array([[0, 0, 0, 1, 1, 1]]), mx.float32)
        for mask in masks.values():
            assert bool(mx.all(mx.isfinite(mask)))

    def test_fully_padded_rows_do_not_produce_nan(self):
        tower = Gemma4TextTower(tiny_config())
        mx.eval(tower.parameters())
        states = tower.all_hidden_states(
            mx.array([[1, 1, 2, 3]]), mx.array([[0, 0, 1, 1]]), eval_every=0
        )
        mx.eval(states[-1])
        assert bool(mx.all(mx.isfinite(states[-1])))

    def test_per_layer_input_variant_builds_and_runs(self):
        """Small Gemma 4 variants add per-layer input embeddings; 12B does not."""
        config = tiny_config(hidden_size_per_layer_input=4, vocab_size_per_layer_input=100)
        tower = Gemma4TextTower(config)
        names = param_names(tower)
        assert "embed_tokens_per_layer.weight" in names
        assert "per_layer_model_projection.weight" in names
        states = tower.all_hidden_states(mx.array([[1, 2, 3]]), eval_every=0)
        mx.eval(states[-1])
        assert bool(mx.all(mx.isfinite(states[-1])))

    def test_twelve_b_shape_has_no_per_layer_inputs(self):
        """Regression guard: the 12B reference must not grow those tensors."""
        config = Gemma4TextConfig.from_dict(dict(GEMMA4_12B_REFERENCE) | {"num_hidden_layers": 2})
        names = param_names(Gemma4TextTower(config))
        assert not any("per_layer" in n for n in names)


# ---------------------------------------------------------------------------
# Sanitize
# ---------------------------------------------------------------------------


class TestSanitize:
    @pytest.mark.parametrize(
        "prefix",
        ["", "model.", "language_model.", "model.language_model.", "language_model.model."],
    )
    def test_every_nesting_lands_on_the_same_names(self, prefix):
        config = tiny_config()
        out = sanitize_weights({f"{prefix}layers.0.mlp.gate_proj.weight": mx.zeros((4, 4))}, config)
        assert set(out) == {"layers.0.mlp.gate_proj.weight"}

    def test_vision_and_audio_towers_are_dropped(self):
        """``gemma4_unified`` packs them alongside; loading them costs GB for nothing."""
        config = tiny_config()
        weights = {
            "model.vision_tower.blocks.0.weight": mx.zeros((2, 2)),
            "audio_tower.encoder.weight": mx.zeros((2, 2)),
            "model.vision_embedder.proj.weight": mx.zeros((2, 2)),
            "multi_modal_projector.weight": mx.zeros((2, 2)),
            "model.embed_tokens.weight": mx.zeros((2, 2)),
        }
        assert set(sanitize_weights(weights, config)) == {"embed_tokens.weight"}

    def test_lm_head_is_dropped(self):
        """Text-encoder use never produces logits."""
        assert sanitize_weights({"lm_head.weight": mx.zeros((2, 2))}, tiny_config()) == {}

    def test_kv_shared_layer_projections_are_dropped(self):
        """mlx-lm needed two separate fixes (#1158, #1240) to get this right."""
        config = tiny_config(num_hidden_layers=6, num_kv_shared_layers=2)
        weights = {
            "model.layers.0.self_attn.k_proj.weight": mx.zeros((2, 2)),
            "model.layers.5.self_attn.k_proj.weight": mx.zeros((2, 2)),
            "model.layers.5.self_attn.k_norm.weight": mx.zeros((2,)),
            "model.layers.5.self_attn.q_proj.weight": mx.zeros((2, 2)),
        }
        out = sanitize_weights(weights, config)
        assert set(out) == {
            "layers.0.self_attn.k_proj.weight",
            "layers.5.self_attn.q_proj.weight",
        }

    def test_v_proj_is_dropped_on_k_eq_v_layers_only(self):
        config = tiny_config()
        full_idx = config.layer_types.index(FULL_ATTENTION)
        sliding_idx = config.layer_types.index(SLIDING_ATTENTION)
        weights = {
            f"model.layers.{full_idx}.self_attn.v_proj.weight": mx.zeros((2, 2)),
            f"model.layers.{sliding_idx}.self_attn.v_proj.weight": mx.zeros((2, 2)),
        }
        out = sanitize_weights(weights, config)
        assert set(out) == {f"layers.{sliding_idx}.self_attn.v_proj.weight"}

    def test_packed_moe_experts_are_split(self):
        config = tiny_config(enable_moe_block=True, num_experts=2, top_k_experts=1, moe_intermediate_size=8)
        weights = {
            "model.layers.0.experts.gate_up_proj": mx.zeros((2, 16, 4)),
            "model.layers.0.experts.down_proj": mx.zeros((2, 4, 8)),
        }
        out = sanitize_weights(weights, config)
        assert set(out) == {
            "layers.0.experts.switch_glu.gate_proj.weight",
            "layers.0.experts.switch_glu.up_proj.weight",
            "layers.0.experts.switch_glu.down_proj.weight",
        }
        assert out["layers.0.experts.switch_glu.gate_proj.weight"].shape == (2, 8, 4)

    def test_unrecognised_keys_are_kept_so_load_fails_loudly(self):
        """A silently dropped tensor is the June-2026 mosaic failure mode."""
        out = sanitize_weights({"model.something.unexpected": mx.zeros((2,))}, tiny_config())
        assert "something.unexpected" in out


# ---------------------------------------------------------------------------
# Pack loading — synthetic safetensors with the expected key names
# ---------------------------------------------------------------------------


def write_pack(directory, config: Gemma4TextConfig, *, shards: int = 1, quantize: int | None = None):
    """Materialise a pack in the layout ``quantize_ltx.py`` emits.

    Sharded ``*.safetensors`` + ``model.safetensors.index.json`` + a
    ``config.json`` carrying the ``quantization`` block, exactly as the
    ``ltx-te-gemma`` recipe writes it.
    """
    tower = Gemma4TextTower(config)
    if quantize is not None:
        nn.quantize(
            tower,
            group_size=32,
            bits=quantize,
            class_predicate=lambda path, module: isinstance(module, (nn.Linear, nn.Embedding)),
        )
    mx.eval(tower.parameters())

    # Write with the on-disk nesting a real checkpoint uses.
    weights = {f"model.{name}": value for name, value in tree_flatten(tower.parameters())}

    keys = list(weights)
    per_shard = (len(keys) + shards - 1) // shards
    weight_map = {}
    for index in range(shards):
        chunk = keys[index * per_shard : (index + 1) * per_shard]
        if not chunk:
            continue
        name = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        mx.save_safetensors(str(directory / name), {k: weights[k] for k in chunk})
        weight_map.update(dict.fromkeys(chunk, name))

    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )

    raw = {
        "model_type": "gemma4_unified",
        "text_config": {
            key: getattr(config, key)
            for key in (
                "hidden_size", "num_hidden_layers", "intermediate_size",
                "num_attention_heads", "num_key_value_heads",
                "num_global_key_value_heads", "attention_k_eq_v",
                "head_dim", "global_head_dim", "sliding_window",
                "sliding_window_pattern", "hidden_size_per_layer_input",
                "num_kv_shared_layers", "use_double_wide_mlp", "vocab_size",
            )
        },
    }
    if quantize is not None:
        raw["quantization"] = {"group_size": 32, "bits": quantize}
    (directory / "config.json").write_text(json.dumps(raw))
    return raw


class TestPackLoading:
    def test_bf16_single_shard_round_trip(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.gemma4_pack import load_gemma4_tower

        config = tiny_config()
        write_pack(tmp_path, config)
        tower = load_gemma4_tower(tmp_path)

        assert isinstance(tower, Gemma4TextTower)
        assert len(tower.layers) == config.num_hidden_layers
        states = tower.all_hidden_states(mx.array([[1, 2, 3]]), eval_every=0)
        mx.eval(states[-1])
        assert bool(mx.all(mx.isfinite(states[-1])))

    def test_sharded_pack_loads_through_the_index(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.gemma4_pack import find_shards, load_gemma4_tower

        write_pack(tmp_path, tiny_config(), shards=4)
        assert len(find_shards(tmp_path)) == 4
        assert isinstance(load_gemma4_tower(tmp_path), Gemma4TextTower)

    @pytest.mark.parametrize("bits", [4, 8])
    def test_quantized_pack_loads_at_the_shipped_recipe(self, tmp_path, bits):
        """q4/q8 with embed_tokens quantized — what ``ltx-te-gemma`` produces.

        ``utils.weights.apply_quantization`` would skip ``embed_tokens``: it only
        accepts ``nn.Linear``. Hence the dedicated predicate in the pack loader.
        """
        from ltx_core_mlx.text_encoders.gemma.gemma4_pack import load_gemma4_tower

        write_pack(tmp_path, tiny_config(), shards=2, quantize=bits)
        tower = load_gemma4_tower(tmp_path)

        assert isinstance(tower.embed_tokens, nn.QuantizedEmbedding)
        assert isinstance(tower.layers[0].mlp.gate_proj, nn.QuantizedLinear)
        assert tower.layers[0].mlp.gate_proj.bits == bits
        assert tower.layers[0].mlp.gate_proj.group_size == 32

        states = tower.all_hidden_states(mx.array([[1, 2, 3]]), eval_every=0)
        mx.eval(states[-1])
        assert bool(mx.all(mx.isfinite(states[-1])))

    def test_stale_index_falls_back_to_globbing(self, tmp_path):
        """The 2.3 pack ships an index naming shards that do not exist."""
        from ltx_core_mlx.text_encoders.gemma.gemma4_pack import find_shards

        write_pack(tmp_path, tiny_config())
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a.weight": "does-not-exist.safetensors"}})
        )
        assert [p.name for p in find_shards(tmp_path)] == ["model-00001-of-00001.safetensors"]

    def test_duplicate_key_across_shards_is_refused(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.gemma4_pack import load_shard_weights

        mx.save_safetensors(str(tmp_path / "a.safetensors"), {"model.norm.weight": mx.zeros((4,))})
        mx.save_safetensors(str(tmp_path / "b.safetensors"), {"model.norm.weight": mx.ones((4,))})
        with pytest.raises(ValueError, match="more than one shard"):
            load_shard_weights(tmp_path)

    def test_missing_tensor_fails_loudly(self, tmp_path):
        """A silently absent tensor is a randomly-initialised module."""
        from ltx_core_mlx.text_encoders.gemma.gemma4_pack import load_gemma4_tower

        config = tiny_config()
        write_pack(tmp_path, config)
        shard = tmp_path / "model-00001-of-00001.safetensors"
        weights = mx.load(str(shard))
        del weights["model.layers.0.mlp.gate_proj.weight"]
        mx.save_safetensors(str(shard), weights)

        with pytest.raises(Exception):
            load_gemma4_tower(tmp_path)

    def test_nested_text_config_is_read_from_the_pack(self, tmp_path):
        from ltx_core_mlx.text_encoders.gemma.gemma4_pack import read_config
        from ltx_core_mlx.text_encoders.gemma.gemma4 import Gemma4TextConfig as Cfg

        config = tiny_config()
        write_pack(tmp_path, config)
        assert Cfg.from_dict(read_config(tmp_path)).hidden_size == config.hidden_size


# ---------------------------------------------------------------------------
# The seam: what the loader resolves, and what it still refuses
# ---------------------------------------------------------------------------


class TestLoaderSeam:
    def test_gemma4_now_resolves_instead_of_raising(self):
        from ltx_core_mlx.text_encoders.gemma.loader import TextEncoderSpec, resolve_text_tower

        spec = TextEncoderSpec(
            model_type="gemma4_unified",
            architecture="gemma4",
            hidden_size=3840,
            num_hidden_layers=48,
            generation=(2, 5),
        )
        assert resolve_text_tower(spec) == "gemma4"

    def test_gemma4_never_resolves_to_a_gemma3_tower(self):
        """The refusal this port was built around, restated as an assertion."""
        from ltx_core_mlx.text_encoders.gemma.loader import TextEncoderSpec, resolve_text_tower

        spec = TextEncoderSpec("gemma4_unified", "gemma4", 3840, 48, (2, 5))
        assert "gemma3" not in resolve_text_tower(spec)

    def test_unknown_model_type_still_refuses(self):
        from ltx_core_mlx.text_encoders.gemma.loader import TextEncoderSpec, resolve_text_tower

        spec = TextEncoderSpec("llama9", "llama9", 4096, 32, (2, 3))
        with pytest.raises(NotImplementedError, match="Unsupported text-encoder"):
            resolve_text_tower(spec)

    def test_gemma3_path_is_untouched(self):
        from ltx_core_mlx.text_encoders.gemma.loader import TextEncoderSpec, resolve_text_tower

        spec = TextEncoderSpec("gemma3", "gemma3", 3840, 48, (2, 3))
        assert resolve_text_tower(spec) == "gemma3"

    def test_projection_width_matches_the_vendored_tower(self):
        """The connector's input width and the tower must agree by construction."""
        from ltx_core_mlx.text_encoders.gemma.loader import TextEncoderSpec

        spec = TextEncoderSpec("gemma4_unified", "gemma4", 3840, 48, (2, 5))
        config = Gemma4TextConfig.from_dict(dict(GEMMA4_12B_REFERENCE))
        assert spec.projection_input_dim == config.hidden_size * (config.num_hidden_layers + 1)
        assert spec.projection_input_dim == 188160
