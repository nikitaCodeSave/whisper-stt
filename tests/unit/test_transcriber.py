"""Tests for stt.core.transcriber — Sprint 2 RED tests."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from stt.core.transcriber import Transcriber, TranscriberConfig, _map_confidence
from stt.exceptions import GpuError, ModelError

# ---------------------------------------------------------------------------
# _map_confidence
# ---------------------------------------------------------------------------

class TestMapConfidence:
    def test_negative_logprob(self) -> None:
        result = _map_confidence(-0.5)
        assert result == pytest.approx(math.exp(-0.5), abs=1e-4)

    def test_zero_logprob(self) -> None:
        result = _map_confidence(0.0)
        assert result == pytest.approx(1.0)

    def test_very_negative_logprob(self) -> None:
        result = _map_confidence(-10.0)
        assert result >= 0.0
        assert result < 0.01

    def test_positive_logprob_clamped(self) -> None:
        result = _map_confidence(1.0)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Transcriber lifecycle
# ---------------------------------------------------------------------------

class TestTranscriberLifecycle:
    def test_transcribe_without_load_raises(self) -> None:
        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        with pytest.raises(RuntimeError):
            t.transcribe("/fake/audio.wav")

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_load_then_transcribe_works(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 2.5
        mock_seg.text = " Hello world"
        mock_seg.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_info.language = "ru"
        mock_info.language_probability = 0.95
        mock_info.duration = 2.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_whisper_cls.return_value = mock_model

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")
        assert isinstance(result, list)
        assert len(result) == 1

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_unload_model_sets_none(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        t.unload_model()
        assert t._model is None

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_unload_calls_empty_cache_when_cuda(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        t.unload_model()

        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_unload_skips_empty_cache_no_cuda(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_whisper_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cpu", compute_type="int8")
        t = Transcriber(config)
        t.load_model()
        mock_torch.cuda.is_available.return_value = False
        t.unload_model()

        mock_torch.cuda.empty_cache.assert_not_called()


# ---------------------------------------------------------------------------
# Transcriber.transcribe() output
# ---------------------------------------------------------------------------

class TestTranscriberTranscribe:
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_returns_segments_ordered_by_start(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True

        seg1 = MagicMock()
        seg1.start = 0.0
        seg1.end = 2.0
        seg1.text = " First"
        seg1.avg_logprob = -0.2

        seg2 = MagicMock()
        seg2.start = 2.5
        seg2.end = 5.0
        seg2.text = " Second"
        seg2.avg_logprob = -0.4

        mock_info = MagicMock()
        mock_info.language = "ru"
        mock_info.language_probability = 0.95
        mock_info.duration = 5.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), mock_info)
        mock_whisper_cls.return_value = mock_model

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")

        assert len(result) == 2
        assert result[0].start <= result[1].start

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_segments_have_no_speaker(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 2.0
        mock_seg.text = " Test"
        mock_seg.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_info.language = "ru"
        mock_info.language_probability = 0.95
        mock_info.duration = 2.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_whisper_cls.return_value = mock_model

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")

        for seg in result:
            assert seg.speaker is None

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_confidence_in_range(self, mock_torch: MagicMock, mock_whisper_cls: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 2.0
        mock_seg.text = " Test"
        mock_seg.avg_logprob = -0.5

        mock_info = MagicMock()
        mock_info.language = "ru"
        mock_info.language_probability = 0.95
        mock_info.duration = 2.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_whisper_cls.return_value = mock_model

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")

        for seg in result:
            assert 0.0 <= seg.confidence <= 1.0


# ---------------------------------------------------------------------------
# model_dir → download_root
# ---------------------------------------------------------------------------

class TestTranscriberModelDir:
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_model_dir_passed_as_download_root(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()

        config = TranscriberConfig(
            model_size="large-v3", device="cuda",
            compute_type="float16", model_dir="/tmp/my_models",
        )
        t = Transcriber(config)
        t.load_model()

        call_kwargs = mock_whisper_cls.call_args
        assert "download_root" in call_kwargs.kwargs

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_no_model_dir_no_download_root(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()

        config = TranscriberConfig(
            model_size="large-v3", device="cuda", compute_type="float16",
        )
        t = Transcriber(config)
        t.load_model()

        call_kwargs = mock_whisper_cls.call_args
        assert "download_root" not in (call_kwargs.kwargs or {})


# ---------------------------------------------------------------------------
# Language passthrough
# ---------------------------------------------------------------------------

class TestTranscriberLanguage:
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_default_language_is_ru(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 2.0
        mock_seg.text = " Test"
        mock_seg.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_whisper_cls.return_value = mock_model

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        t.transcribe("/fake/audio.wav")

        mock_model.transcribe.assert_called_once_with("/fake/audio.wav", language="ru")

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_language_en_passed_through(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True

        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 2.0
        mock_seg.text = " Test"
        mock_seg.avg_logprob = -0.3

        mock_info = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
        mock_whisper_cls.return_value = mock_model

        config = TranscriberConfig(
            model_size="large-v3", device="cuda",
            compute_type="float16", language="en",
        )
        t = Transcriber(config)
        t.load_model()
        t.transcribe("/fake/audio.wav")

        mock_model.transcribe.assert_called_once_with("/fake/audio.wav", language="en")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestTranscriberErrors:
    @patch("stt.core.transcriber.torch")
    def test_cuda_unavailable_raises_gpu_error(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = False

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        with pytest.raises(GpuError):
            t.load_model()

    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_model_not_found_raises_model_error(
        self, mock_torch: MagicMock, mock_whisper_cls: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.side_effect = Exception("Model not found")

        config = TranscriberConfig(
            model_size="nonexistent-model", device="cuda", compute_type="float16",
        )
        t = Transcriber(config)
        with pytest.raises(ModelError):
            t.load_model()
