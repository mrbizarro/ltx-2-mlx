"""+ltx25.8 contracts.

1. Conditioning-image CRF follows the checkpoint generation (Lightricks
   ``LTX_2_4_IMAGE_CRF = 18`` from 2.4 on, 33 before), an explicit CRF wins,
   and every image-taking pipeline call site passes the pipeline's own value.
2. IC-LoRA two-stage: Stage 2 denoises through the CLEAN reloaded transformer,
   not the Stage 1 X0Model that still wraps the adapter.
3. Distilled retake (dgrauet/ltx-2-mlx#204 recipe): the distilled transformer,
   the checkpoint's own distilled table, positive prompt only, plain Euler;
   extend refuses the mode.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import ltx_pipelines_mlx
from ltx_pipelines_mlx import retake as retake_mod
from ltx_pipelines_mlx.retake import RetakePipeline
from ltx_pipelines_mlx.utils import _orchestration, media_io
from ltx_pipelines_mlx.utils.args import ImageConditioningInput
from ltx_pipelines_mlx.utils.samplers import DenoiseOutput

PKG = Path(ltx_pipelines_mlx.__file__).parent


# --------------------------------------------------------------------------- 1


@pytest.fixture(autouse=True)
def _no_crf_env(monkeypatch):
    monkeypatch.delenv(media_io.IMAGE_CRF_ENV_VAR, raising=False)


@pytest.mark.parametrize(
    "version,crf",
    [((), 33), (None, 33), ((2, 3), 33), ((2, 3, 1), 33), ((2, 4), 18), ((2, 5, 0), 18), ((3,), 18)],
)
def test_crf_by_generation(version, crf):
    assert media_io.image_crf_for_version(version) == crf


def test_crf_env_override(monkeypatch):
    monkeypatch.setenv(media_io.IMAGE_CRF_ENV_VAR, "0")
    assert media_io.image_crf_for_version((2, 5)) == 0
    monkeypatch.setenv(media_io.IMAGE_CRF_ENV_VAR, "nonsense")
    assert media_io.image_crf_for_version((2, 5)) == 18


def test_image_input_leaves_crf_to_the_checkpoint():
    assert ImageConditioningInput("a.png", 0, 1.0).crf is None
    assert ImageConditioningInput(*("a.png", 0, 1.0)).crf is None


class _FakeEncoder:
    def encode(self, x):
        b, _c, f, h, w = x.shape
        return mx.zeros((b, 128, f, max(1, h // 32), max(1, w // 32)))


def _crfs_used(monkeypatch, images, default_crf):
    seen = []

    def fake_load(path, h, w, crf):
        seen.append(crf)
        return mx.zeros((1, 3, h, w))

    monkeypatch.setattr(media_io, "load_image_and_preprocess", fake_load)
    _orchestration.combined_image_conditionings(
        images,
        enc_h=64,
        enc_w=64,
        spatial_dims=(2, 2, 2),
        video_encoder=_FakeEncoder(),
        frame_rate=24.0,
        default_crf=default_crf,
    )
    return seen


def test_unset_crf_takes_the_pipeline_default(monkeypatch):
    imgs = [ImageConditioningInput("a.png", 0, 1.0)]
    assert _crfs_used(monkeypatch, imgs, 18) == [18]
    assert _crfs_used(monkeypatch, imgs, None) == [33]


def test_explicit_crf_wins_including_zero(monkeypatch):
    imgs = [ImageConditioningInput("a.png", 0, 1.0, 0), ImageConditioningInput("b.png", 9, 1.0, 25)]
    assert _crfs_used(monkeypatch, imgs, 18) == [0, 25]


def test_every_image_call_site_passes_the_pipeline_crf():
    """A call site that forgets `default_crf` silently stays on 33 for 2.5."""
    for path in sorted(PKG.glob("*.py")):
        src = path.read_text()
        for m in re.finditer(r"combined_image_conditionings\(", src):
            if src[max(0, m.start() - 40) : m.start()].rstrip().endswith("import"):
                continue
            call = src[m.start() : src.index(")\n", m.start())]
            assert "default_crf=self._default_image_crf()" in call, f"{path.name}: {call[:80]}"


class _Cfg(SimpleNamespace):
    pass


def _stub_pipeline(tmp_path, *, dit_version=None, config_version=None):
    from ltx_pipelines_mlx._base import BasePipeline

    pipe = BasePipeline.__new__(BasePipeline)
    pipe.dit = None if dit_version is None else SimpleNamespace(config=_Cfg(model_version=dit_version))
    pipe.model_dir = tmp_path
    if config_version is not None:
        (tmp_path / "config.json").write_text(json.dumps({"model_version": config_version}))
    return pipe


def test_pipeline_crf_from_loaded_dit(tmp_path):
    assert _stub_pipeline(tmp_path, dit_version=(2, 5, 0))._default_image_crf() == 18
    assert _stub_pipeline(tmp_path, dit_version=(2, 3))._default_image_crf() == 33


def test_pipeline_crf_before_load_reads_the_pack(tmp_path):
    assert _stub_pipeline(tmp_path, config_version="2.5.0")._default_image_crf() == 18


def test_pipeline_crf_unknown_pack_is_23(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert _stub_pipeline(tmp_path)._default_image_crf() == 33


# --------------------------------------------------------------------------- 2


def test_ic_stage2_runs_the_reloaded_clean_transformer():
    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline

    tree = ast.parse(inspect.getsource(ICLoraPipeline.generate).lstrip().replace("\n    ", "\n", 0))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = node.body
        calls = [i for i, st in enumerate(body) if "_reload_clean_transformer" in ast.unparse(st)]
        if not calls:
            continue
        after = [ast.unparse(st) for st in body[calls[0] + 1 :]]
        assert any(s.replace(" ", "") == "x0_model=X0Model(self.dit)" for s in after), after
        found = True
    assert found, "no _reload_clean_transformer() branch found in ICLoraPipeline.generate"


# --------------------------------------------------------------------------- 3


def _stub_retake(version=(2, 5, 0), distilled=True):
    from ltx_core_mlx.components.patchifiers import AudioPatchifier, VideoLatentPatchifier

    pipe = RetakePipeline.__new__(RetakePipeline)
    pipe.distilled = distilled
    pipe.low_memory = False
    pipe.dit = SimpleNamespace(config=_Cfg(model_version=version))
    pipe.video_patchifier = VideoLatentPatchifier()
    pipe.audio_patchifier = AudioPatchifier()
    pipe._pre_denoise_flush = lambda *a, **k: None
    return pipe


def test_distilled_retake_schedule_is_the_checkpoint_table():
    from ltx_pipelines_mlx.scheduler import resolve_distilled_schedule

    for version in ((2, 5, 0), (2, 3)):
        pipe = _stub_retake(version)
        full = [float(x) for x in resolve_distilled_schedule(version)[0]]
        assert pipe.distilled_sigmas(None) == full
        assert pipe.distilled_sigmas(8) == full
        assert pipe.distilled_sigmas(30) == full  # the dev-lane default must not raise
        thin = pipe.distilled_sigmas(4)
        assert len(thin) == 5 and thin[0] == 1.0 and thin[-1] == 0.0


def test_distilled_retake_runs_positive_only_on_plain_euler(monkeypatch):
    pipe = _stub_retake()
    calls = {}

    def no_negative(*a, **k):
        raise AssertionError("distilled retake must not encode a negative prompt")

    pipe._encode_text_with_negative = no_negative
    pipe._load_dev_transformer = no_negative
    pipe._encode_positive_prompt = lambda prompt: (mx.zeros((1, 4, 8)), mx.zeros((1, 4, 8)))

    def fake_loop(**kw):
        calls.update(kw)
        return DenoiseOutput(video_latent=kw["video_state"].latent, audio_latent=kw["audio_state"].latent)

    def refuse_guided(**kw):
        raise AssertionError("guided loop on the distilled lane")

    monkeypatch.setattr(retake_mod, "denoise_loop", fake_loop)
    monkeypatch.setattr(retake_mod, "guided_denoise_loop", refuse_guided)
    F, H, W = 3, 2, 2
    video = mx.zeros((1, 128, F, H, W))
    audio = mx.zeros((1, 8, 10, 16))
    v, a = pipe.retake(
        "p",
        video,
        audio,
        1,
        2,
        height=H * 32,
        width=W * 32,
        num_frames=17,
        frame_rate=24.0,
        num_steps=8,
        regenerate_audio=False,
    )
    assert v.shape == video.shape
    assert calls["sigmas"][0] == 1.0 and calls["sigmas"][-1] == 0.0
    assert calls.get("diffusion_step") is None  # deterministic Euler, as upstream
    # Keep-sound: every audio token preserved; only latent frame 1 regenerated.
    assert float(calls["audio_state"].denoise_mask.sum()) == 0.0
    vm = calls["video_state"].denoise_mask.reshape(F, H * W)
    assert vm[1].min().item() == 1.0 and vm[0].max().item() == 0.0 and vm[2].max().item() == 0.0


def test_distilled_extend_appends_on_plain_euler(monkeypatch):
    pipe = _stub_retake()
    calls = {}

    def no_negative(*a, **k):
        raise AssertionError("distilled extend must not encode a negative prompt or load the dev model")

    pipe._encode_text_with_negative = no_negative
    pipe._load_dev_transformer = no_negative
    pipe._encode_positive_prompt = lambda prompt: (mx.zeros((1, 4, 8)), mx.zeros((1, 4, 8)))

    def fake_loop(**kw):
        calls.update(kw)
        return DenoiseOutput(video_latent=kw["video_state"].latent, audio_latent=kw["audio_state"].latent)

    def refuse_guided(**kw):
        raise AssertionError("guided loop on the distilled lane")

    monkeypatch.setattr(retake_mod, "denoise_loop", fake_loop)
    monkeypatch.setattr(retake_mod, "guided_denoise_loop", refuse_guided)
    F, H, W = 3, 2, 2
    v, _a = pipe.extend(
        "p",
        mx.zeros((1, 128, F, H, W)),
        mx.zeros((1, 8, 13, 16)),
        2,
        height=H * 32,
        width=W * 32,
        num_frames=17,
        frame_rate=24.0,
        num_steps=8,
    )
    assert v.shape == (1, 128, F + 2, H, W)
    assert calls["sigmas"][0] == 1.0 and calls["sigmas"][-1] == 0.0
    vm = calls["video_state"].denoise_mask.reshape(F + 2, H * W)
    assert vm[:F].max().item() == 0.0 and vm[F:].min().item() == 1.0  # source kept, new frames generated


def test_constructor_default_keeps_the_dev_lane():
    sig = inspect.signature(RetakePipeline.__init__)
    assert sig.parameters["distilled"].default is False
