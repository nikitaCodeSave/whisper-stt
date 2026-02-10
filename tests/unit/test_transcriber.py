"""Tests for stt.core.transcriber — sequential and batched modes."""

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

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_load_then_transcribe_works(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
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

        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")
        assert isinstance(result, list)
        assert len(result) == 1

        # Default mode uses model.transcribe(), not batched
        mock_model.transcribe.assert_called_once()
        mock_batched_cls.return_value.transcribe.assert_not_called()

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_unload_model_sets_none(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()
        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        t.unload_model()
        assert t._model is None
        assert t._batched is None

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_unload_calls_cleanup_gpu_memory(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()
        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        t.unload_model()

        mock_cleanup.assert_called_once_with("transcriber_unload")

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_unload_calls_cleanup_regardless_of_cuda(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_whisper_cls.return_value = MagicMock()
        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cpu", compute_type="int8")
        t = Transcriber(config)
        t.load_model()
        mock_torch.cuda.is_available.return_value = False
        t.unload_model()

        mock_cleanup.assert_called_once_with("transcriber_unload")


# ---------------------------------------------------------------------------
# Transcriber.transcribe() output
# ---------------------------------------------------------------------------

class TestTranscriberTranscribe:
    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_returns_segments_ordered_by_start(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
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

        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")

        assert len(result) == 2
        assert result[0].start <= result[1].start

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_segments_have_no_speaker(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
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

        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")

        for seg in result:
            assert seg.speaker is None

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_confidence_in_range(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
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

        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        result = t.transcribe("/fake/audio.wav")

        for seg in result:
            assert 0.0 <= seg.confidence <= 1.0


# ---------------------------------------------------------------------------
# model_dir -> download_root
# ---------------------------------------------------------------------------

class TestTranscriberModelDir:
    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_model_dir_passed_as_download_root(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()
        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(
            model_size="large-v3", device="cuda",
            compute_type="float16", model_dir="/tmp/my_models",
        )
        t = Transcriber(config)
        t.load_model()

        call_kwargs = mock_whisper_cls.call_args
        assert "download_root" in call_kwargs.kwargs

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_no_model_dir_no_download_root(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()
        mock_batched_cls.return_value = MagicMock()

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
    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_default_language_is_ru(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
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

        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(model_size="large-v3", device="cuda", compute_type="float16")
        t = Transcriber(config)
        t.load_model()
        t.transcribe("/fake/audio.wav")

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["language"] == "ru"

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_language_en_passed_through(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
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

        mock_batched_cls.return_value = MagicMock()

        config = TranscriberConfig(
            model_size="large-v3", device="cuda",
            compute_type="float16", language="en",
        )
        t = Transcriber(config)
        t.load_model()
        t.transcribe("/fake/audio.wav")

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["language"] == "en"


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

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_model_not_found_raises_model_error(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.side_effect = Exception("Model not found")

        config = TranscriberConfig(
            model_size="nonexistent-model", device="cuda", compute_type="float16",
        )
        t = Transcriber(config)
        with pytest.raises(ModelError):
            t.load_model()


# ---------------------------------------------------------------------------
# BatchedInferencePipeline specific tests
# ---------------------------------------------------------------------------

class TestTranscriberBatchedPipeline:
    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_load_creates_batched_pipeline(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model
        mock_batched_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig())
        t.load_model()
        mock_batched_cls.assert_called_once_with(model=mock_model)

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_vad_filter_passed(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_batched = MagicMock()
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_batched.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_batched_cls.return_value = mock_batched
        mock_whisper_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig(vad_filter=True, use_batched=True))
        t.load_model()
        t.transcribe("/fake.wav")
        call_kwargs = mock_batched.transcribe.call_args.kwargs
        assert call_kwargs["vad_filter"] is True

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_batch_size_passed(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_batched = MagicMock()
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_batched.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_batched_cls.return_value = mock_batched
        mock_whisper_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig(batch_size=16, use_batched=True))
        t.load_model()
        t.transcribe("/fake.wav")
        call_kwargs = mock_batched.transcribe.call_args.kwargs
        assert call_kwargs["batch_size"] == 16

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_hallucination_threshold_passed(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_batched = MagicMock()
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_batched.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_batched_cls.return_value = mock_batched
        mock_whisper_cls.return_value = MagicMock()

        t = Transcriber(
            TranscriberConfig(hallucination_silence_threshold=3.0, use_batched=True),
        )
        t.load_model()
        t.transcribe("/fake.wav")
        call_kwargs = mock_batched.transcribe.call_args.kwargs
        assert call_kwargs["hallucination_silence_threshold"] == 3.0

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_condition_on_previous_text_passed(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_batched = MagicMock()
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_batched.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_batched_cls.return_value = mock_batched
        mock_whisper_cls.return_value = MagicMock()

        t = Transcriber(
            TranscriberConfig(condition_on_previous_text=True, use_batched=True),
        )
        t.load_model()
        t.transcribe("/fake.wav")
        call_kwargs = mock_batched.transcribe.call_args.kwargs
        assert call_kwargs["condition_on_previous_text"] is True

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_unload_clears_batched(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_cls.return_value = MagicMock()
        mock_batched_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig())
        t.load_model()
        assert t._batched is not None
        t.unload_model()
        assert t._batched is None
        assert t._model is None


# ---------------------------------------------------------------------------
# Sequential vs Batched routing
# ---------------------------------------------------------------------------

class TestTranscriberRouting:
    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_default_uses_model_transcribe(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_whisper_cls.return_value = mock_model
        mock_batched = MagicMock()
        mock_batched_cls.return_value = mock_batched

        t = Transcriber(TranscriberConfig())
        t.load_model()
        t.transcribe("/fake.wav")

        mock_model.transcribe.assert_called_once()
        mock_batched.transcribe.assert_not_called()

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_use_batched_uses_batched_pipeline(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model
        mock_batched = MagicMock()
        mock_batched.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_batched_cls.return_value = mock_batched

        t = Transcriber(TranscriberConfig(use_batched=True))
        t.load_model()
        t.transcribe("/fake.wav")

        mock_batched.transcribe.assert_called_once()
        mock_model.transcribe.assert_not_called()

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_batched_passes_without_timestamps_false(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_batched = MagicMock()
        mock_batched.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_batched_cls.return_value = mock_batched
        mock_whisper_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig(use_batched=True))
        t.load_model()
        t.transcribe("/fake.wav")

        call_kwargs = mock_batched.transcribe.call_args.kwargs
        assert call_kwargs["without_timestamps"] is False

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_model_transcribe_passes_vad_params(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_whisper_cls.return_value = mock_model
        mock_batched_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig(vad_filter=True))
        t.load_model()
        t.transcribe("/fake.wav")

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["vad_filter"] is True
        assert call_kwargs["vad_parameters"] == {"min_silence_duration_ms": 500}

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_model_transcribe_passes_hallucination_threshold(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_whisper_cls.return_value = mock_model
        mock_batched_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig(hallucination_silence_threshold=3.0))
        t.load_model()
        t.transcribe("/fake.wav")

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["hallucination_silence_threshold"] == 3.0

    @patch("stt.core.transcriber.cleanup_gpu_memory")
    @patch("stt.core.transcriber.BatchedInferencePipeline")
    @patch("stt.core.transcriber.WhisperModel")
    @patch("stt.core.transcriber.torch")
    def test_model_transcribe_passes_condition_on_previous_text(
        self,
        mock_torch: MagicMock,
        mock_whisper_cls: MagicMock,
        mock_batched_cls: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_seg = MagicMock(start=0.0, end=2.0, text=" X", avg_logprob=-0.3)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([mock_seg]), MagicMock())
        mock_whisper_cls.return_value = mock_model
        mock_batched_cls.return_value = MagicMock()

        t = Transcriber(TranscriberConfig(condition_on_previous_text=True))
        t.load_model()
        t.transcribe("/fake.wav")

        call_kwargs = mock_model.transcribe.call_args.kwargs
        assert call_kwargs["condition_on_previous_text"] is True
