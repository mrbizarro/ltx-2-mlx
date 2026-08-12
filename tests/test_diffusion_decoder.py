"""Parity tests for the MLX LTX-2.5 diffusion VAE decoder.

Methodology is this package's established one: the reference is transcribed longhand
into NumPy from the *published* implementations, independently of the MLX code, so a
shared misreading cannot pass both. Sources transcribed:

* ``comfy/ldm/lightricks/vae/na_diffusion_decoder.py`` @ Comfy-Org/ComfyUI ``57ce8e1a``
* ``ltx_core/model/video_vae/transformer/fallback_na/eager.py`` @ Lightricks/LTX-2
  ``v1.2.0`` (the vendor's own CPU statement of what NATTEN's ``na3d`` computes)
* ``ltx_core/model/video_vae/transformer/rope_math.py`` @ same tag

Three things get their own tests because each fails *silently*:

1. the float64 RoPE table (ComfyUI #15512 / PR #15516),
2. NATTEN's inward window shift at grid borders,
3. the patchify channel order ``(c, w_sub, h_sub)``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.utils
import numpy as np
import pytest

from ltx_core_mlx.model.video_vae.diffusion_decoder import (
    DiffusionVideoDecoder,
    patchify_spatial_cl,
    timestep_embedding,
    unpatchify_spatial_cl,
)
from ltx_core_mlx.model.video_vae.na3d import (
    apply_abs_rope,
    default_rope_dim_split,
    na3d,
    rope_inv_freqs,
    window_bounds,
)

# ---------------------------------------------------------------------------
# NumPy reference (transcribed by hand from the sources above)
# ---------------------------------------------------------------------------


def np_rope_inv_freqs(dim: int, base: float = 10000.0) -> np.ndarray:
    exponents = np.arange(0, dim, 2, dtype=np.float64) / dim
    return 1.0 / np.power(np.float64(base), exponents)


def np_window_bounds(length: int, kernel: int):
    kernel = min(kernel, length)
    lo = length - kernel
    half = kernel // 2
    starts = [min(max(i - half, 0), lo) for i in range(length)]
    return starts, [s + kernel for s in starts]


def np_na3d(q, k, v, kernel_size, scale=None):
    """Brute force: build each query's window explicitly. O(N * prod(kernel))."""
    b, t, h, w, nh, hd = q.shape
    if scale is None:
        scale = hd**-0.5
    bt = np_window_bounds(t, kernel_size[0])
    bh = np_window_bounds(h, kernel_size[1])
    bw = np_window_bounds(w, kernel_size[2])
    out = np.zeros_like(v, dtype=np.float64)
    q = q.astype(np.float64) * scale
    k64, v64 = k.astype(np.float64), v.astype(np.float64)
    for ti in range(t):
        ts, te = bt[0][ti], bt[1][ti]
        for hi in range(h):
            hs, he = bh[0][hi], bh[1][hi]
            for wi in range(w):
                ws, we = bw[0][wi], bw[1][wi]
                kk = k64[:, ts:te, hs:he, ws:we].reshape(b, -1, nh, hd)
                vv = v64[:, ts:te, hs:he, ws:we].reshape(b, -1, nh, hd)
                qq = q[:, ti, hi, wi]  # (b, nh, hd)
                scores = np.einsum("bnd,bknd->bnk", qq, kk)
                scores = scores - scores.max(axis=-1, keepdims=True)
                p = np.exp(scores)
                p = p / p.sum(axis=-1, keepdims=True)
                out[:, ti, hi, wi] = np.einsum("bnk,bknd->bnd", p, vv)
    return out


def np_rot_axis(chunk, positions, inv, axis):
    pairs = chunk.reshape(*chunk.shape[:-1], chunk.shape[-1] // 2, 2).astype(np.float64)
    xe, xo = pairs[..., 0], pairs[..., 1]
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = positions.shape[0]
    ang = (positions[:, None] * inv[None, :]).reshape(shape)
    c, s = np.cos(ang), np.sin(ang)
    return np.stack([xe * c - xo * s, xe * s + xo * c], axis=-1).reshape(chunk.shape)


def np_apply_abs_rope(x, split, invs):
    d_t, d_h, _ = split
    t, h, w = x.shape[1], x.shape[2], x.shape[3]
    xt = np_rot_axis(x[..., :d_t], np.arange(t, dtype=np.float64), invs[0], 1)
    xh = np_rot_axis(x[..., d_t : d_t + d_h], np.arange(h, dtype=np.float64), invs[1], 2)
    xw = np_rot_axis(x[..., d_t + d_h :], np.arange(w, dtype=np.float64), invs[2], 3)
    return np.concatenate([xt, xh, xw], axis=-1)


def np_rms_norm(x, weight, eps=1e-6):
    x = x.astype(np.float64)
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * weight.astype(np.float64)


def np_silu(x):
    return x / (1.0 + np.exp(-x))


def np_linear(x, w, b=None):
    y = x.astype(np.float64) @ w.astype(np.float64).T
    return y if b is None else y + b.astype(np.float64)


class NpDecoder:
    """Longhand transcription of ``NADiffusionDecoder`` against a flat weight dict."""

    def __init__(self, p, cfg):
        self.p = p
        self.cfg = cfg
        self.patch = cfg["patch_size"]
        self.head_dim = cfg["head_dim"]
        self.split = default_rope_dim_split(self.head_dim)
        self.invs = tuple(np_rope_inv_freqs(d).astype(np.float32).astype(np.float64) for d in self.split)
        self.temporal_upscale = math.prod(s[0] for s, _ in cfg["upsamples"])
        self.trailing = (cfg["stage_kernels"][0][0] // 2) * 2

    def attn(self, x, pre, kernel):
        w = self.p[f"{pre}.qkv.weight"]
        dim = w.shape[0] // 3
        nh = dim // self.head_dim
        qkv = np_linear(x, w, self.p[f"{pre}.qkv.bias"])
        b, t, h, ww = x.shape[:4]
        shape = (b, t, h, ww, nh, self.head_dim)
        q = qkv[..., :dim].reshape(shape)
        k = qkv[..., dim : 2 * dim].reshape(shape)
        v = qkv[..., 2 * dim :].reshape(shape)
        q = np_rms_norm(q, self.p[f"{pre}.q_norm.weight"] * (self.head_dim**-0.5))
        k = np_rms_norm(k, self.p[f"{pre}.k_norm.weight"])
        q = np_apply_abs_rope(q, self.split, self.invs)
        k = np_apply_abs_rope(k, self.split, self.invs)
        o = np_na3d(q, k, v, kernel, scale=1.0).reshape(b, t, h, ww, dim)
        return np_linear(o, self.p[f"{pre}.proj.weight"], self.p[f"{pre}.proj.bias"])

    def mlp(self, x, pre):
        g = np_silu(np_linear(x, self.p[f"{pre}.w_gate.weight"]))
        u = np_linear(x, self.p[f"{pre}.w_up.weight"])
        return np_linear(g * u, self.p[f"{pre}.w_down.weight"])

    def na_block(self, x, pre, kernel):
        x = x + self.attn(np_rms_norm(x, self.p[f"{pre}.norm1.weight"]), f"{pre}.attn", kernel)
        return x + self.mlp(np_rms_norm(x, self.p[f"{pre}.norm2.weight"]), f"{pre}.mlp")

    def upsample(self, x, i, drop_leading):
        stride, _ = self.cfg["upsamples"][i]
        p1, p2, p3 = stride
        y = np_linear(x, self.p[f"upsamples.{i}.proj.weight"], self.p[f"upsamples.{i}.proj.bias"])
        b, t, h, w = x.shape[:4]
        c = y.shape[-1] // (p1 * p2 * p3)
        y = y.reshape(b, t, h, w, c, p1, p2, p3).transpose(0, 1, 5, 2, 6, 3, 7, 4)
        y = y.reshape(b, t * p1, h * p2, w * p3, c)
        if p1 == 2 and drop_leading:
            y = y[:, 1:]
        return y

    def pre_diffusion(self, z, drop_leading=True, pad_trailing=True):
        n = self.trailing if pad_trailing else 0
        if n > 0:
            z = np.concatenate([z, np.repeat(z[:, -1:], n, axis=1)], axis=1)
        x = np_linear(z, self.p["conv_in.weight"], self.p["conv_in.bias"])
        for i in range(len(self.cfg["stage_channels"]) - 1):
            for d in range(self.cfg["stage_depths"][i]):
                x = self.na_block(x, f"det_stages.{i}.{d}", self.cfg["stage_kernels"][i])
            x = self.upsample(x, i, drop_leading)
        if n > 0:
            x = x[:, : -(n * self.temporal_upscale)]
        return x

    def diff_step(self, ctx, x_t, t):
        x = np_patchify(x_t, self.patch)
        x = np_linear(x, self.p["conv_in_x_t.weight"], self.p["conv_in_x_t.bias"])
        emb = np_timestep_embedding(np.array([t * self.cfg["timestep_scale_multiplier"]]), 256)
        emb = np_linear(emb, self.p["t_embedder.mlp.0.weight"], self.p["t_embedder.mlp.0.bias"])
        emb = np_linear(np_silu(emb), self.p["t_embedder.mlp.2.weight"], self.p["t_embedder.mlp.2.bias"])
        h = np_linear(np_silu(emb), self.p["shared_adaln.proj.weight"], self.p["shared_adaln.proj.bias"])
        dim = h.shape[-1] // 7
        mod = [h[:, i * dim : (i + 1) * dim][:, None, None, None, :] for i in range(7)]
        for i in range(self.cfg["stage_depths"][-1]):
            pre = f"diff_blocks.{i}"
            table = self.p[f"{pre}.scale_shift_table"]
            m = [mod[j] + table[j].reshape(1, 1, 1, 1, -1) for j in range(7)]
            s_msa, sh_msa, _, s_mlp, sh_mlp, _, _ = m
            x = x + np_linear(ctx, self.p[f"{pre}.context_proj.weight"], self.p[f"{pre}.context_proj.bias"])
            pre_x = np_rms_norm(x, self.p[f"{pre}.norm1.weight"]) * (1.0 + s_msa) + sh_msa
            x = x + self.attn(pre_x, f"{pre}.attn", self.cfg["stage5_kernel"])
            pre_x = np_rms_norm(x, self.p[f"{pre}.norm2.weight"]) * (1.0 + s_mlp) + sh_mlp
            x = x + self.mlp(pre_x, f"{pre}.mlp")
        x = np_rms_norm(x, self.p["norm_out.weight"])
        x = np_linear(x, self.p["conv_out.weight"], self.p["conv_out.bias"])
        return np_unpatchify(x, self.patch, self.cfg["out_channels"])


def np_patchify(x, p):
    b, t, h, w, c = x.shape
    y = x.reshape(b, t, h // p, p, w // p, p, c).transpose(0, 1, 2, 4, 6, 5, 3)
    return y.reshape(b, t, h // p, w // p, c * p * p)


def np_unpatchify(x, p, c):
    b, t, h, w, _ = x.shape
    y = x.reshape(b, t, h, w, c, p, p).transpose(0, 1, 2, 6, 3, 5, 4)
    return y.reshape(b, t, h * p, w * p, c)


def np_timestep_embedding(timesteps, dim, max_period=10000.0):
    half = dim // 2
    exponent = -math.log(max_period) * np.arange(half, dtype=np.float64) / half
    emb = timesteps.astype(np.float64)[:, None] * np.exp(exponent)[None, :]
    return np.concatenate([np.cos(emb), np.sin(emb)], axis=-1)


# ---------------------------------------------------------------------------
# 1. The float64 RoPE table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [4, 6, 16, 24, 64])
def test_rope_table_matches_a_float64_reference(dim):
    """MLX has no float64. Build the table in numpy f64 and cast once, like the vendor."""
    got = np.asarray(rope_inv_freqs(dim))
    want = np_rope_inv_freqs(dim).astype(np.float32)
    np.testing.assert_array_equal(got, want)


def test_naive_float32_rope_table_is_measurably_wrong():
    """The trap from ComfyUI #15512 / PR #15516: computing the table natively in f32
    does not crash on MLX -- it returns *different* frequencies. If this ever starts
    passing at equality, someone has replaced the f64 construction with an f32 one and
    the decoder is silently degraded."""
    dim = 24
    exponents_f32 = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    naive = (1.0 / np.power(np.float32(10000.0), exponents_f32)).astype(np.float32)
    correct = np.asarray(rope_inv_freqs(dim))
    assert not np.array_equal(naive, correct), "f32 and f64 tables agreed exactly -- check the reference"
    assert np.abs(naive - correct).max() > 0.0


def test_rope_split_for_the_checkpoint_head_dim():
    assert default_rope_dim_split(64) == (16, 24, 24)


def test_abs_rope_matches_longhand_numpy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 4, 5, 2, 16)).astype(np.float32)
    split = default_rope_dim_split(16)
    invs_mx = tuple(rope_inv_freqs(d) for d in split)
    invs_np = tuple(np.asarray(i).astype(np.float64) for i in invs_mx)
    got = np.asarray(apply_abs_rope(mx.array(x), split, invs_mx))
    want = np_apply_abs_rope(x, split, invs_np)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. NATTEN semantics
# ---------------------------------------------------------------------------


def test_window_shifts_inward_at_borders_instead_of_shrinking():
    """Every query gets exactly ``kernel`` keys; the border ones slide in.

    A clamp-and-mask implementation would give the first query a 2-wide window here.
    Both would look plausible; only one is what the checkpoint was trained against."""
    starts, ends = window_bounds(7, 3)
    assert [e - s for s, e in zip(starts, ends)] == [3] * 7
    assert starts[0] == 0 and ends[0] == 3
    assert starts[-1] == 4 and ends[-1] == 7
    # Kernel wider than the axis collapses to full attention over that axis.
    assert window_bounds(3, 11) == ((0, 0, 0), (3, 3, 3))


@pytest.mark.parametrize(
    ("dims", "kernel"),
    [
        ((3, 4, 5), (3, 3, 3)),
        ((5, 5, 5), (3, 5, 5)),
        ((2, 6, 7), (11, 5, 5)),  # kernel wider than an axis
        ((7, 3, 3), (3, 7, 7)),
    ],
)
def test_na3d_matches_brute_force_gather(dims, kernel):
    rng = np.random.default_rng(1)
    t, h, w = dims
    shape = (1, t, h, w, 2, 16)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    got = np.asarray(na3d(mx.array(q), mx.array(k), mx.array(v), kernel))
    want = np_na3d(q, k, v, kernel)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)


def test_na3d_tiling_does_not_change_the_answer():
    """The score budget only controls how the work is split; the result must not move."""
    rng = np.random.default_rng(2)
    shape = (1, 6, 6, 6, 2, 16)
    q, k, v = (rng.standard_normal(shape).astype(np.float32) for _ in range(3))
    big = np.asarray(na3d(mx.array(q), mx.array(k), mx.array(v), (3, 3, 3), score_budget=2**24))
    small = np.asarray(na3d(mx.array(q), mx.array(k), mx.array(v), (3, 3, 3), score_budget=64))
    np.testing.assert_allclose(big, small, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Patchify channel order and the timestep embedding
# ---------------------------------------------------------------------------


def test_patchify_channel_order_is_c_then_w_then_h():
    """``b c (h q) (w r) -> b (c r q) h w``: the W subdivision is the *outer* of the two
    spatial sub-indices. Swapping them keeps every shape and scrambles the picture."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 2, 4, 6, 3)).astype(np.float32)
    got = np.asarray(patchify_spatial_cl(mx.array(x), 2))
    np.testing.assert_allclose(got, np_patchify(x, 2), rtol=0, atol=0)


def test_unpatchify_inverts_patchify():
    rng = np.random.default_rng(4)
    x = mx.array(rng.standard_normal((1, 2, 4, 6, 3)).astype(np.float32))
    back = unpatchify_spatial_cl(patchify_spatial_cl(x, 2), 2, 3)
    np.testing.assert_allclose(np.asarray(back), np.asarray(x), rtol=0, atol=0)


def test_timestep_embedding_flips_sin_and_cos():
    t = mx.array([1000.0, 1.0])
    got = np.asarray(timestep_embedding(t, 16))
    want = np_timestep_embedding(np.array([1000.0, 1.0]), 16)
    # The reference here is float64 while both MLX and torch run this in float32; at
    # t=1000 the highest-frequency entries differ by ~3e-5 for that reason alone.
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    # The flip is the part that fails silently: cos block first, sin block second.
    half = 8
    assert got[0, 0] == pytest.approx(math.cos(1000.0), abs=1e-4)
    assert got[0, half] == pytest.approx(math.sin(1000.0), abs=1e-4)


# ---------------------------------------------------------------------------
# 4. Whole-decoder parity at tiny dims
# ---------------------------------------------------------------------------

# The stage widths are not free: ``LinearPixelShuffleUpsample`` makes
# ``stage_channels[i+1] == stage_channels[i] // reduction[i]``, so with the real
# reductions (2, 2, 1, 2) the widths must fall as c, c/2, c/4, c/4, c/8 -- exactly as
# the checkpoint's 2048, 1024, 512, 512, 256 do. A tiny config that ignores that
# builds a decoder whose stages cannot be chained.
TINY_CFG = {
    "in_channels": 8,
    "out_channels": 3,
    "patch_size": 2,
    "head_dim": 16,
    "stage_channels": [128, 64, 32, 32, 16],
    "stage_depths": [1, 1, 1, 1, 2],
    "stage_kernels": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 2]],
    "stage5_kernel": [3, 3, 3],
    "t_emb_dim": 384,
    "timestep_scale_multiplier": 1000.0,
    "default_num_inference_steps": 1,
    "model_output_type": "x0",
}


def _tiny_model_and_params():
    mx.random.seed(7)
    model = DiffusionVideoDecoder(TINY_CFG)
    flat = dict(mlx.utils.tree_flatten(model.parameters()))
    randomized = {}
    for key, value in flat.items():
        if value.ndim == 0:
            randomized[key] = value
        else:
            randomized[key] = mx.random.normal(value.shape) * 0.2
    model.update(mlx.utils.tree_unflatten(list(randomized.items())))
    np_params = {k[len("decoder.") :]: np.asarray(v) for k, v in randomized.items() if k.startswith("decoder.")}
    return model, np_params


def test_pre_diffusion_stages_match_numpy():
    """Stages 1-4 are deterministic, so they can be compared exactly -- this is where a
    wrong pixel-shuffle axis order or a wrong ``drop_leading_frame`` would show."""
    model, p = _tiny_model_and_params()
    rng = np.random.default_rng(5)
    z = rng.standard_normal((1, 2, 3, 3, 8)).astype(np.float32)
    got = np.asarray(model.decoder.forward_pre_diffusion(mx.array(z)))
    want = NpDecoder(p, TINY_CFG).pre_diffusion(z)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=3e-3, atol=3e-3)


def test_pre_diffusion_output_geometry_is_8x_temporal_32x_spatial():
    model, _ = _tiny_model_and_params()
    z = mx.zeros((1, 4, 3, 3, 8))
    ctx = model.decoder.forward_pre_diffusion(z)
    # 8k+1 temporal contract: 4 latent frames -> 4*8 - 7 = 25 output frames.
    assert ctx.shape[1] == 4 * 8 - 7
    # Context is at pixel/patch_size resolution.
    assert ctx.shape[2] == 3 * model.spatial_upscale // TINY_CFG["patch_size"]


def test_full_decoder_forward_matches_numpy():
    """The diffusion step is deterministic given the noise, so feed both the same noise."""
    model, p = _tiny_model_and_params()
    rng = np.random.default_rng(6)
    z = rng.standard_normal((1, 2, 3, 3, 8)).astype(np.float32)
    ctx_mx = model.decoder.forward_pre_diffusion(mx.array(z))
    ctx_np = NpDecoder(p, TINY_CFG).pre_diffusion(z)
    x_t = rng.standard_normal((1, ctx_np.shape[1], ctx_np.shape[2] * 2, ctx_np.shape[3] * 2, 3)).astype(np.float32)
    got = np.asarray(model.decoder.forward_diff_step(ctx_mx, mx.array(x_t), mx.array([1.0])))
    want = NpDecoder(p, TINY_CFG).diff_step(ctx_np, x_t, 1.0)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# 5. Wrapper contract and tiling
# ---------------------------------------------------------------------------


def test_decode_returns_bcfhw_with_the_conv_decoders_geometry():
    model, _ = _tiny_model_and_params()
    latent = mx.zeros((1, 8, 2, 3, 3))  # (B, C, F, H, W)
    out = model.decode(latent)
    assert out.shape == (1, 3, 2 * 8 - 7, 3 * model.spatial_upscale, 3 * model.spatial_upscale)


def test_temporal_tiling_covers_every_frame_and_is_off_by_default_when_small(monkeypatch):
    model, _ = _tiny_model_and_params()
    assert model.plan_temporal_tiles(16, 14, 24) == [(0, 16)]
    monkeypatch.setenv("LTX2_DIFFVAE_BUDGET_GB", "0.001")
    tiles = model.plan_temporal_tiles(16, 14, 24)
    assert len(tiles) > 1
    assert tiles[0][0] == 0 and tiles[-1][1] == 16
    covered = set()
    for a, b in tiles:
        covered |= set(range(a, b))
    assert covered == set(range(16))
    # Consecutive tiles must overlap, or the cross-fade has nothing to fade.
    for (a0, a1), (b0, _b1) in zip(tiles, tiles[1:]):
        assert b0 < a1 and b0 > a0


def test_tiled_decode_matches_untiled_shape(monkeypatch):
    model, _ = _tiny_model_and_params()
    latent = mx.random.normal((1, 8, 4, 3, 3)) * 0.5
    untiled = model.decode(latent)
    monkeypatch.setenv("LTX2_DIFFVAE_BUDGET_GB", "0.0000001")
    tiled = model.decode(latent)
    assert tiled.shape == untiled.shape


def test_type_emb_is_carried_but_never_read():
    """The checkpoint ships ``decoder.type_emb`` (128,) and neither reference
    implementation references it. It is loaded so a strict load passes and nothing is
    dropped unrecorded -- but changing it must not change a single output pixel."""
    model, _ = _tiny_model_and_params()
    latent = mx.random.normal((1, 8, 2, 3, 3)) * 0.5
    before = np.asarray(model.decode(latent))
    model.decoder.type_emb = mx.random.normal(model.decoder.type_emb.shape) * 100.0
    after = np.asarray(model.decode(latent))
    np.testing.assert_array_equal(before, after)


def test_strict_load_rejects_a_pack_missing_a_module():
    """A non-strict load is how a randomly-initialised module ships. Pin the refusal."""
    from ltx_core_mlx.model.video_vae.diffusion_decoder import load_diffusion_decoder

    model = DiffusionVideoDecoder(TINY_CFG)
    weights = {f"vae_decoder_diffusion.{k}": v for k, v in mlx.utils.tree_flatten(model.parameters())}
    weights.pop("vae_decoder_diffusion.decoder.conv_out.weight")
    with pytest.raises(ValueError):
        load_diffusion_decoder(weights, TINY_CFG)
