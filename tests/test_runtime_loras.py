"""The unfused runtime-LoRA branch — exactness, dispatch, and the loud refusals.

The claim under test is narrow and checkable: **applying a LoRA as
``base(x) + s·B(A x)`` is exact at any bit width, while fusing it into a
quantized weight is not.** Everything else here guards the ways that could be
true in the small and still fail in the large — a wrong module skipped in
silence, a state-dict path renamed so a later fusion misses, an ``auto`` mode
that picks fusion on a q4 pack.

Numbers this file pins, measured on the real packs in
``notes/ltx25_lora_investigation.md`` §3.6 with ``bizarrotrn_v2``
(rank 32, ``‖B@A‖/‖W‖ ≈ 0.08``) and reproduced here at tiny dims:

======================  ==================================
arm                     delta destroyed ``‖eff−D‖/‖D‖``
======================  ==================================
unfused, any bit width  ~2e-7  (float round-off only)
fused into bf16         ~2 %
fused into int8         ~7-10 %
fused into int4         **~94 %**
======================  ==================================
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_core_mlx.loader.fuse_loras import apply_loras
from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core_mlx.loader.runtime_loras import (
    LoRALinear,
    LoRAQuantizedLinear,
    attach_loras,
    model_has_quantized_linears,
    resolve_lora_mode,
)

GROUP_SIZE = 64
#: The real ratio a rank-32 character LoRA carries against LTX's attention
#: weights. The whole quantization story is a function of this number.
DELTA_OVER_WEIGHT = 0.08


def _norm(a: mx.array) -> float:
    return float(mx.sqrt(mx.sum(a.astype(mx.float32) ** 2)).item())


def _rel(a: mx.array, b: mx.array) -> float:
    return _norm(a - b) / _norm(b)


def _lora_pair(in_features: int, out_features: int, rank: int, base: mx.array) -> tuple[mx.array, mx.array]:
    """A/B scaled so ``‖B@A‖/‖base‖`` matches the real character-LoRA ratio."""
    a = mx.random.normal((rank, in_features))
    b = mx.random.normal((out_features, rank))
    scale = DELTA_OVER_WEIGHT * _norm(base) / _norm(b @ a)
    return a * scale, b


def _quantized_linear(in_features: int, out_features: int, bits: int) -> tuple[nn.QuantizedLinear, mx.array]:
    """A QuantizedLinear plus the dequantized weight the model will actually run."""
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight = linear.weight.astype(mx.float32)
    linear.bias = linear.bias.astype(mx.float32)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=GROUP_SIZE, bits=bits)
    effective = mx.dequantize(
        quantized["weight"],
        scales=quantized["scales"],
        biases=quantized.get("biases"),
        group_size=GROUP_SIZE,
        bits=bits,
    ).astype(mx.float32)
    return quantized, effective


# ---------------------------------------------------------------------------
# 1. Exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [4, 8])
def test_unfused_on_quantized_matches_the_float_reference(bits):
    """``base(x) + s·B(A x)`` == ``(W̃ + s·BA)x + bias`` computed in float32.

    ``W̃`` is the *dequantized* weight — what the quantized matmul actually
    multiplies by. The reference is the ideal the fusion path is trying and
    failing to reach on a quantized pack; the runtime branch reaches it at
    float round-off, and the bit width does not appear in the error.
    """
    mx.random.seed(11)
    in_features, out_features, rank = 256, 128, 8
    quantized, effective = _quantized_linear(in_features, out_features, bits)
    a, b = _lora_pair(in_features, out_features, rank, effective)
    strength = 0.75

    adapter = LoRAQuantizedLinear(quantized, a, b, strength)
    x = mx.random.normal((4, in_features))

    reference = x.astype(mx.float32) @ (effective + strength * (b @ a).astype(mx.float32)).T
    reference = reference + quantized["bias"].astype(mx.float32)

    assert _rel(adapter(x), reference) < 1e-5


def test_unfused_on_float_matches_the_float_reference():
    """Same claim for the bf16/float target, so one mode means one thing everywhere."""
    mx.random.seed(12)
    in_features, out_features, rank = 256, 128, 8
    linear = nn.Linear(in_features, out_features, bias=True)
    a, b = _lora_pair(in_features, out_features, rank, linear["weight"])

    adapter = LoRALinear(linear, a, b, 1.0)
    x = mx.random.normal((4, in_features))
    reference = x.astype(mx.float32) @ (linear["weight"].astype(mx.float32) + (b @ a)).T
    reference = reference + linear["bias"].astype(mx.float32)

    assert _rel(adapter(x), reference) < 1e-5


@pytest.mark.parametrize("bits", [4, 8])
def test_measured_delta_survival_unfused_beats_bf16_fusion_beats_quantized_fusion(bits):
    """The investigation's own measurement, re-run at tiny dims on three arms.

    Method (``lora_quant_survival.py`` §3.6): take the delta the LoRA intends,
    ``D = B@A``, and compare it against the *effective* delta each application
    path actually gives the model. The unfused branch's effective delta is ``D``
    **by construction**, so it is measured functionally — from the layer's own
    outputs — rather than asserted from the algebra.
    """
    mx.random.seed(13)
    in_features, out_features, rank = 512, 512, 32
    quantized, effective = _quantized_linear(in_features, out_features, bits)
    a, b = _lora_pair(in_features, out_features, rank, effective)
    delta = (b @ a).astype(mx.float32)
    assert abs(_norm(delta) / _norm(effective) - DELTA_OVER_WEIGHT) < 1e-3

    # Arm 1 — unfused, measured on real activations: (y_lora - y_base) vs D·x.
    adapter = LoRAQuantizedLinear(quantized, a, b, 1.0)
    x = mx.random.normal((8, in_features))
    unfused_lost = _rel(adapter(x) - quantized(x), x.astype(mx.float32) @ delta.T)

    # Arm 2 — fused into bf16: one rounding of W + D to bfloat16.
    bf16_effective = (effective + delta).astype(mx.bfloat16).astype(mx.float32) - effective
    bf16_lost = _rel(bf16_effective, delta)

    # Arm 3 — fused into the quantized weight: dequantize, add, re-quantize.
    packed, scales, biases = mx.quantize(effective + delta, group_size=GROUP_SIZE, bits=bits)
    quantized_effective = (
        mx.dequantize(packed, scales=scales, biases=biases, group_size=GROUP_SIZE, bits=bits).astype(mx.float32)
        - effective
    )
    quantized_lost = _rel(quantized_effective, delta)

    assert unfused_lost < 1e-5, "the runtime branch must carry the delta whole"
    assert unfused_lost < bf16_lost < quantized_lost
    # bf16 fusion is the honest reference the unfused branch must match, not beat:
    # ~2 % lost at this delta size (bfloat16's relative ULP is 3.9e-3).
    assert 0.005 < bf16_lost < 0.05
    if bits == 4:
        # The headline: at int4 the fused delta is ~95 % gone. This is the
        # number two investigations were spent chasing.
        assert quantized_lost > 0.5
    else:
        assert quantized_lost > 2 * bf16_lost


def test_adapters_are_narrowed_to_the_layers_compute_dtype_and_stay_accurate():
    """An fp32 LoRA on a bf16 layer is narrowed — and the delta still survives.

    ``bizarrotrn_v2`` ships as float32 (856 MB). Against a bfloat16 pack that is
    double the memory, fp32 matmuls for the low-rank hops, and no fused addmm
    epilogue, for accuracy the forward cannot use. What the narrowing costs is
    bounded here: the delta's relative error must stay far below what fusing
    into bf16 (2.1 %) would have cost, let alone int4 (94 %).
    """
    mx.random.seed(23)
    in_features, out_features, rank = 512, 512, 32
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight = linear.weight.astype(mx.bfloat16)
    quantized = nn.QuantizedLinear.from_linear(linear, group_size=GROUP_SIZE, bits=4)
    assert quantized["scales"].dtype == mx.bfloat16

    effective = mx.dequantize(
        quantized["weight"],
        scales=quantized["scales"],
        biases=quantized.get("biases"),
        group_size=GROUP_SIZE,
        bits=4,
    ).astype(mx.float32)
    a, b = _lora_pair(in_features, out_features, rank, effective)
    adapter = LoRAQuantizedLinear(*_narrowed(quantized, a, b), 1.0)

    assert adapter["lora_a"].dtype == mx.bfloat16
    assert adapter["lora_b"].dtype == mx.bfloat16

    x = mx.random.normal((8, in_features)).astype(mx.bfloat16)
    want = x.astype(mx.float32) @ (b @ a).astype(mx.float32).T

    # The delta the narrowed adapter actually contributes, straight from its own
    # arrays — not recovered by differencing two bfloat16 outputs, which would
    # amplify the layer's rounding by |y|/|delta| (~12x here) and measure the
    # subtraction rather than the adapter.
    narrowed_delta = ((x @ adapter["lora_a"].T) @ adapter["lora_b"].T).astype(mx.float32)
    assert _rel(narrowed_delta, want) < 0.01, "narrowing must cost far less than any fusion would"

    # And as a share of what the layer outputs, which is what a render sees.
    y = adapter(x).astype(mx.float32)
    assert _norm(narrowed_delta - want) / _norm(y) < 1e-3


def _narrowed(base, a, b):
    """``(base, a, b)`` with the adapters narrowed exactly as ``attach_loras`` does."""
    from ltx_core_mlx.loader.runtime_loras import _match_dtype

    return (base, *_match_dtype(a, b, base))


def test_zero_strength_is_a_strict_no_op():
    """A zero-strength LoRA must be bit-identical to no LoRA at all."""
    mx.random.seed(14)
    quantized, effective = _quantized_linear(128, 64, 4)
    a, b = _lora_pair(128, 64, 8, effective)
    adapter = LoRAQuantizedLinear(quantized, a, b, 0.0)
    x = mx.random.normal((3, 128))
    assert mx.array_equal(adapter(x), quantized(x))


# ---------------------------------------------------------------------------
# 2. Attachment on a model tree
# ---------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def __call__(self, x):
        return self.to_out(self.to_q(x))


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn1 = _Attn(dim)

    def __call__(self, x):
        return self.attn1(x)


class _Tiny(nn.Module):
    """A two-block stand-in with the same shape of name as the real DiT."""

    def __init__(self, dim: int = 128, blocks: int = 2):
        super().__init__()
        self.transformer_blocks = [_Block(dim) for _ in range(blocks)]

    def __call__(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x


def _tiny_lora(model: _Tiny, rank: int = 8, targets: tuple[str, ...] = ("attn1.to_q", "attn1.to_out")):
    """A LoRA state dict already in post-rename (model-key) form."""
    sd: dict[str, mx.array] = {}
    for index, block in enumerate(model.transformer_blocks):
        for target in targets:
            module = block
            for part in target.split("."):
                module = module[part]
            in_features = int(module["scales"].shape[-1]) * GROUP_SIZE if "scales" in module else int(
                module["weight"].shape[-1]
            )
            out_features = int(module["weight"].shape[0])
            a = mx.random.normal((rank, in_features)) * 0.01
            b = mx.random.normal((out_features, rank)) * 0.01
            sd[f"transformer_blocks.{index}.{target}.lora_A.weight"] = a
            sd[f"transformer_blocks.{index}.{target}.lora_B.weight"] = b
    return sd


def _spec(sd: dict[str, mx.array], strength: float = 1.0) -> LoraStateDictWithStrength:
    return LoraStateDictWithStrength(state_dict=StateDict(sd=sd, size=0, dtype=set()), strength=strength)


def test_attach_wraps_every_target_and_keeps_state_dict_paths():
    """The adapters must not rename ``…to_q.weight``.

    A wrapper module would have moved it to ``…to_q.base.weight``, and the two
    later fusion sites (``_fuse_distilled_lora``, ``ICLoraPipeline._fuse_loras``)
    match ``X.lora_A.weight`` against ``X.weight`` — so the distilled LoRA would
    have found no target on every attention projection and been dropped in
    silence. That is issue #52's failure mode, rebuilt.
    """
    from mlx.utils import tree_flatten

    mx.random.seed(15)
    model = _Tiny()
    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
    before = {key for key, _ in tree_flatten(model.parameters())}

    report = attach_loras(model, [_spec(_tiny_lora(model))], verbose=False)

    after = {key for key, _ in tree_flatten(model.parameters())}
    assert len(report.applied) == 4
    assert report.quantized_targets == 4
    assert not report.skipped
    assert before <= after, "attaching must not move or drop an existing parameter"
    assert after - before == {
        f"transformer_blocks.{i}.attn1.{t}.lora_{ab}"
        for i in (0, 1)
        for t in ("to_q", "to_out")
        for ab in ("a", "b")
    }
    assert isinstance(model.transformer_blocks[0].attn1.to_q, LoRAQuantizedLinear)


def test_a_later_weight_fusion_still_finds_the_attached_modules():
    """Regression for the stage-2 distilled LoRA on top of a runtime character LoRA.

    Two-stage renders fuse the rank-384 distilled LoRA into the live DiT after
    stage 1. With a character LoRA already attached, that fusion must still land
    on the base weights underneath it.
    """
    from mlx.utils import tree_flatten

    mx.random.seed(16)
    model = _Tiny()
    nn.quantize(model, group_size=GROUP_SIZE, bits=8)
    attach_loras(model, [_spec(_tiny_lora(model))], verbose=False)

    flat = {key: value for key, value in tree_flatten(model.parameters()) if isinstance(value, mx.array)}
    second = _tiny_lora(model, rank=4)
    fused = apply_loras(StateDict(sd=flat, size=0, dtype=set()), [_spec(second)], verbose=False)

    # Every base weight changed (the second LoRA landed), and the runtime
    # branch's own parameters survived the round trip untouched.
    for key in flat:
        if key.endswith(".weight"):
            assert not mx.array_equal(fused.sd[key], flat[key]), f"{key} did not receive the fused delta"
        if key.endswith(".lora_a") or key.endswith(".lora_b"):
            assert mx.array_equal(fused.sd[key], flat[key])
    model.load_weights(list(fused.sd.items()))  # strict: the key sets must still agree


def test_attach_equals_float_fusion_on_the_same_quantized_base():
    """End to end on a model: unfused == fusing in float, no re-quantization."""
    mx.random.seed(17)
    model = _Tiny()
    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
    lora = _tiny_lora(model)

    reference = _Tiny()
    for index, block in enumerate(model.transformer_blocks):
        for target in ("to_q", "to_out"):
            module = block.attn1[target]
            dequantized = mx.dequantize(
                module["weight"],
                scales=module["scales"],
                biases=module.get("biases"),
                group_size=GROUP_SIZE,
                bits=4,
            ).astype(mx.float32)
            a = lora[f"transformer_blocks.{index}.attn1.{target}.lora_A.weight"]
            b = lora[f"transformer_blocks.{index}.attn1.{target}.lora_B.weight"]
            reference.transformer_blocks[index].attn1[target].weight = dequantized + (b @ a).astype(mx.float32)

    attach_loras(model, [_spec(lora)], verbose=False)
    x = mx.random.normal((2, 128))
    assert _rel(model(x), reference(x)) < 1e-5


def test_two_loras_on_one_module_concatenate_on_the_rank_axis():
    """``s₁B₁A₁ + s₂B₂A₂`` from a single rank-(r₁+r₂) branch, no weight temporary."""
    mx.random.seed(18)
    in_features, out_features = 128, 128
    _, effective = _quantized_linear(in_features, out_features, 4)
    a1, b1 = _lora_pair(in_features, out_features, 4, effective)
    a2, b2 = _lora_pair(in_features, out_features, 6, effective)

    model = _Tiny(dim=128, blocks=1)
    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
    target = "transformer_blocks.0.attn1.to_q"
    first = {f"{target}.lora_A.weight": a1, f"{target}.lora_B.weight": b1}
    second = {f"{target}.lora_A.weight": a2, f"{target}.lora_B.weight": b2}

    base = model.transformer_blocks[0].attn1.to_q
    x = mx.random.normal((3, in_features))
    baseline = base(x)
    report = attach_loras(model, [_spec(first, 0.5), _spec(second, 1.5)], verbose=False)

    assert len(report.applied) == 1
    assert report.applied[0].rank == 10
    got = model.transformer_blocks[0].attn1.to_q(x) - baseline
    want = 0.5 * (x @ a1.T) @ b1.T + 1.5 * (x @ a2.T) @ b2.T
    assert _rel(got, want) < 1e-5


# ---------------------------------------------------------------------------
# 3. Refusals — nothing may be dropped in silence
# ---------------------------------------------------------------------------


def test_a_missing_module_is_skipped_with_a_reason():
    mx.random.seed(19)
    model = _Tiny(blocks=1)
    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
    sd = {
        "transformer_blocks.0.attn1.to_nowhere.lora_A.weight": mx.zeros((4, 128)),
        "transformer_blocks.0.attn1.to_nowhere.lora_B.weight": mx.zeros((128, 4)),
        "transformer_blocks.9.attn1.to_q.lora_A.weight": mx.zeros((4, 128)),
        "transformer_blocks.9.attn1.to_q.lora_B.weight": mx.zeros((128, 4)),
    }
    report = attach_loras(model, [_spec(sd)], verbose=False)
    assert not report.applied
    assert len(report.skipped) == 2
    assert all("no matching module" in module.reason for module in report.skipped)


def test_a_shape_mismatch_is_skipped_with_the_logical_dims_in_the_message():
    """The packed-vs-logical trap: ``weight`` on a q4 layer has in/8 columns.

    Comparing a LoRA's logical width against packed storage skipped all 208
    modules on the H3 Q8 DiT — silently, with the render still 'working'.
    ``scales`` carries the unpacked width, so a genuine mismatch is reported in
    logical dims and a legitimate LoRA is never rejected for being unpacked.
    """
    mx.random.seed(20)
    model = _Tiny(blocks=1)
    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
    sd = {
        "transformer_blocks.0.attn1.to_q.lora_A.weight": mx.zeros((4, 64)),
        "transformer_blocks.0.attn1.to_q.lora_B.weight": mx.zeros((128, 4)),
    }
    report = attach_loras(model, [_spec(sd)], verbose=False)
    assert not report.applied
    assert "shape mismatch: base [128, 128]" in report.skipped[0].reason


def test_zero_matches_shouts(capsys):
    mx.random.seed(21)
    model = _Tiny(blocks=1)
    sd = {
        "somewhere.else.lora_A.weight": mx.zeros((4, 8)),
        "somewhere.else.lora_B.weight": mx.zeros((8, 4)),
    }
    attach_loras(model, [_spec(sd)], verbose=False)
    assert "0 modules attached" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 4. Mode resolution
# ---------------------------------------------------------------------------


def test_auto_picks_unfused_on_a_quantized_model_and_fuse_on_a_float_one():
    model = _Tiny(blocks=1)
    assert not model_has_quantized_linears(model)
    assert resolve_lora_mode("auto", model) == "fuse"

    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
    assert model_has_quantized_linears(model)
    assert resolve_lora_mode("auto", model) == "unfused"


def test_an_explicit_mode_is_honoured_and_an_unknown_one_raises():
    model = _Tiny(blocks=1)
    nn.quantize(model, group_size=GROUP_SIZE, bits=4)
    assert resolve_lora_mode("fuse", model) == "fuse"
    assert resolve_lora_mode("unfused", model) == "unfused"
    with pytest.raises(ValueError, match="lora mode must be one of"):
        resolve_lora_mode("merge", model)


# ---------------------------------------------------------------------------
# 5. The fusion path's new warnings
# ---------------------------------------------------------------------------


def _fusion_fixture(bits: int):
    """A one-module quantized state dict plus a LoRA that targets it."""
    mx.random.seed(22)
    quantized, effective = _quantized_linear(256, 128, bits)
    a, b = _lora_pair(256, 128, 16, effective)
    model_sd = StateDict(
        sd={
            "layer.weight": quantized["weight"],
            "layer.scales": quantized["scales"],
            "layer.biases": quantized["biases"],
        },
        size=0,
        dtype=set(),
    )
    lora = _spec({"layer.lora_A.weight": a, "layer.lora_B.weight": b})
    return model_sd, lora


def test_fusing_into_int4_warns_with_the_measured_loss(capsys):
    """The warning is a measurement of this pack + this LoRA, not a constant."""
    model_sd, lora = _fusion_fixture(4)
    apply_loras(model_sd, [lora])
    err = capsys.readouterr().err
    assert "QUANTIZED modules (int4)" in err
    assert "--lora-mode unfused" in err
    percent = float(err.split("MEASURED ")[1].split("%")[0])
    assert percent > 50.0


def test_quantized_ok_acknowledges_the_loss_and_silences_the_warning(capsys):
    model_sd, lora = _fusion_fixture(4)
    apply_loras(model_sd, [lora], quantized_ok=True)
    assert "MEASURED" not in capsys.readouterr().err


def test_fusing_into_int8_reports_but_does_not_shout(capsys):
    """~7-10 % at int8 is below the shout threshold — reported, not alarmed."""
    model_sd, lora = _fusion_fixture(8)
    apply_loras(model_sd, [lora])
    err = capsys.readouterr().err
    assert "MEASURED" not in err
    assert "re-quantization cost" in err


def test_a_lora_that_matches_nothing_is_reported_not_silent(capsys):
    """Issue #52: a prefix mismatch made every render a LoRA-free render."""
    model_sd = StateDict(sd={"layer.weight": mx.zeros((8, 8))}, size=0, dtype=set())
    lora = _spec({"elsewhere.lora_A.weight": mx.zeros((2, 8)), "elsewhere.lora_B.weight": mx.zeros((8, 2))})
    apply_loras(model_sd, [lora])
    assert "0 modules matched" in capsys.readouterr().err


def test_float_fusion_stays_quiet_about_quantization(capsys):
    model_sd = StateDict(sd={"layer.weight": mx.random.normal((8, 8))}, size=0, dtype=set())
    lora = _spec({"layer.lora_A.weight": mx.zeros((2, 8)), "layer.lora_B.weight": mx.zeros((8, 2))})
    apply_loras(model_sd, [lora])
    err = capsys.readouterr().err
    assert "QUANTIZED" not in err
    assert "1 modules (0 quantized, 1 float)" in err


# ---------------------------------------------------------------------------
# 6. Pipeline dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_stub():
    """The dispatch surface only — no weights, no MLX work."""

    class _Stub:
        verbose = False
        low_ram_streaming = False
        lora_mode = "auto"

        def _fuse_pending_loras(self, weights, pending):
            self._fuse_spy = (weights, pending)
            return {"__fused__": True}

        def _attach_pending_loras(self, dit, pending):
            self._attach_spy = (dit, pending)

    from ltx_pipelines_mlx._base import BasePipeline

    stub = _Stub()
    stub._resolve_lora_mode = BasePipeline._resolve_lora_mode.__get__(stub)
    return stub


def _load(stub, path="/fake/transformer.safetensors"):
    from ltx_pipelines_mlx._base import BasePipeline

    return BasePipeline._load_transformer_with_optional_streaming(stub, Path(path))


def test_auto_on_a_quantized_pack_takes_the_unfused_path(pipeline_stub):
    """The default must not fuse into q4. This is the whole point."""
    pipeline_stub._pending_loras = [("/fake/lora.safetensors", 1.0)]
    sentinel = MagicMock(name="dit")
    with (
        patch("ltx_pipelines_mlx.utils._orchestration.transformer_pack_is_quantized", return_value=True),
        patch("ltx_pipelines_mlx.utils._orchestration.load_transformer", return_value=sentinel) as orch_load,
    ):
        result = _load(pipeline_stub)

    assert result is sentinel
    orch_load.assert_called_once_with(Path("/fake/transformer.safetensors"), low_ram_streaming=False)
    assert pipeline_stub._attach_spy == (sentinel, [("/fake/lora.safetensors", 1.0)])
    assert not hasattr(pipeline_stub, "_fuse_spy")


def test_auto_on_a_float_pack_takes_the_fusion_path(pipeline_stub):
    """bf16 fusion is exact and free — ``auto`` keeps it there."""
    pipeline_stub._pending_loras = [("/fake/lora.safetensors", 1.0)]
    with (
        patch("ltx_pipelines_mlx.utils._orchestration.transformer_pack_is_quantized", return_value=False),
        patch("ltx_pipelines_mlx._base.load_split_safetensors", return_value={"raw": "weights"}),
        patch("ltx_pipelines_mlx._base.apply_quantization"),
        patch("ltx_pipelines_mlx._base.LTXModel"),
        patch("ltx_pipelines_mlx._base.LTXModelConfig"),
        patch("ltx_pipelines_mlx._base.aggressive_cleanup"),
        patch("ltx_pipelines_mlx._base.mx.eval"),
    ):
        _load(pipeline_stub)

    assert pipeline_stub._fuse_spy[1] == [("/fake/lora.safetensors", 1.0)]
    assert not hasattr(pipeline_stub, "_attach_spy")


def test_unfused_under_low_ram_streaming_refuses(pipeline_stub):
    """Refuse rather than quietly bind-time-fuse what the user asked to keep unfused."""
    pipeline_stub._pending_loras = [("/fake/lora.safetensors", 1.0)]
    pipeline_stub.low_ram_streaming = True
    pipeline_stub.lora_mode = "unfused"
    with pytest.raises(ValueError, match="not supported with low_ram_streaming"):
        _load(pipeline_stub)


def test_an_unknown_pipeline_lora_mode_raises(pipeline_stub):
    pipeline_stub._pending_loras = [("/fake/lora.safetensors", 1.0)]
    pipeline_stub.lora_mode = "merge"
    with pytest.raises(ValueError, match="lora_mode must be one of"):
        _load(pipeline_stub)
