"""H36: the Gemma 3 text tower must build its attention mask in the hidden-state dtype.

A float16 Gemma failed every LTX render with
"[scaled_dot_product_attention] Mask type must promote to output type float16"
because the mask was always bfloat16.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel


class _Attn(nn.Module):
    def __call__(self, h, mask=None, cache=None):
        B, T, D = h.shape
        q = h.reshape(B, T, 1, D).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, mask=mask)
        return out.transpose(0, 2, 1, 3).reshape(B, T, D)


class _Inner(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.embed_tokens.weight = self.embed_tokens.weight.astype(dtype)
        self.layers = [_Attn(), _Attn()]


class _Outer(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.model = _Inner(dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_mask_follows_hidden_dtype(dtype):
    lm = GemmaLanguageModel(architecture="gemma3")
    lm._model = _Outer(dtype)
    ids = mx.array([[0, 0, 3, 4, 5]])
    valid = mx.array([[0, 0, 1, 1, 1]])
    states = lm.get_all_hidden_states(ids, valid)
    mx.eval(states)
    assert states[-1].dtype == dtype
    # Real (non-padding) rows stay finite.
    assert bool(mx.all(mx.isfinite(states[-1][:, 2:].astype(mx.float32))).item())
