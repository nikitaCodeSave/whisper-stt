"""Tests for stt.core.diarizer — Sprint 3 RED tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from stt.core.diarizer import (
    DiarizationResult,
    DiarizationTurn,
    DiarizerConfig,
    PyannoteDiarizer,
)
from stt.exceptions import ModelError

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TestDiarizationTurn:
    def test_fields(self) -> None:
        turn = DiarizationTurn(start=0.0, end=5.0, speaker="SPEAKER_00")
        assert turn.start == 0.0
        assert turn.end == 5.0
        assert turn.speaker == "SPEAKER_00"


class TestDiarizationResult:
    def test_fields(self) -> None:
        turns = [
            DiarizationTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationTurn(start=5.5, end=10.0, speaker="SPEAKER_01"),
        ]
        result = DiarizationResult(turns=turns, num_speakers=2)
        assert len(result.turns) == 2
        assert result.num_speakers == 2


# ---------------------------------------------------------------------------
# PyannoteDiarizer
# ---------------------------------------------------------------------------

class TestPyannoteDiarizerAvailability:
    @patch("stt.core.diarizer.importlib.import_module")
    def test_is_available_true(self, mock_import: MagicMock) -> None:
        mock_import.return_value = MagicMock()
        assert PyannoteDiarizer.is_available() is True

    @patch("stt.core.diarizer.importlib.import_module", side_effect=ImportError)
    def test_is_available_false(self, mock_import: MagicMock) -> None:
        assert PyannoteDiarizer.is_available() is False


class TestPyannoteDiarizerDiarize:
    @patch("stt.core.diarizer.Pipeline")
    def test_diarize_returns_result(self, mock_pipeline_cls: MagicMock) -> None:
        mock_turn1 = MagicMock()
        mock_turn1.start = 0.0
        mock_turn1.end = 5.0

        mock_turn2 = MagicMock()
        mock_turn2.start = 5.5
        mock_turn2.end = 10.0

        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = [
            (mock_turn1, None, "SPEAKER_00"),
            (mock_turn2, None, "SPEAKER_01"),
        ]

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        d.load_model()
        result = d.diarize("/fake/audio.wav")

        assert isinstance(result, DiarizationResult)
        assert len(result.turns) == 2
        assert result.num_speakers == 2

    @patch("stt.core.diarizer.Pipeline")
    def test_passes_num_speakers(self, mock_pipeline_cls: MagicMock) -> None:
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = []

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        config = DiarizerConfig(num_speakers=3)
        d = PyannoteDiarizer(config)
        d.load_model()
        d.diarize("/fake/audio.wav")

        call_kwargs = mock_pipeline.call_args
        assert call_kwargs is not None
        # num_speakers should be passed to the pipeline call
        kw = call_kwargs.kwargs or (
            call_kwargs[1] if len(call_kwargs) > 1 else {}
        )
        assert "num_speakers" in kw

    @patch("stt.core.diarizer.Pipeline")
    def test_passes_min_max_speakers(self, mock_pipeline_cls: MagicMock) -> None:
        mock_annotation = MagicMock()
        mock_annotation.itertracks.return_value = []

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = mock_annotation
        mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

        config = DiarizerConfig(min_speakers=2, max_speakers=5)
        d = PyannoteDiarizer(config)
        d.load_model()
        d.diarize("/fake/audio.wav")

        call_kwargs = mock_pipeline.call_args
        assert call_kwargs is not None
        kw = call_kwargs.kwargs or (
            call_kwargs[1] if len(call_kwargs) > 1 else {}
        )
        assert "min_speakers" in kw
        assert "max_speakers" in kw


class TestPyannoteDiarizerCacheDir:
    @patch("stt.core.diarizer.Pipeline")
    def test_cache_dir_passed_to_from_pretrained(
        self, mock_pipeline_cls: MagicMock,
    ) -> None:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()

        config = DiarizerConfig(cache_dir="/tmp/my_models")
        d = PyannoteDiarizer(config)
        d.load_model()

        call_kwargs = mock_pipeline_cls.from_pretrained.call_args
        assert "cache_dir" in call_kwargs.kwargs

    @patch("stt.core.diarizer.Pipeline")
    def test_no_cache_dir_not_passed(
        self, mock_pipeline_cls: MagicMock,
    ) -> None:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()

        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        d.load_model()

        call_kwargs = mock_pipeline_cls.from_pretrained.call_args
        assert "cache_dir" not in (call_kwargs.kwargs or {})


class TestPyannoteDiarizerHfToken:
    @patch("stt.core.diarizer.Pipeline")
    def test_hf_token_passed_when_set(
        self, mock_pipeline_cls: MagicMock,
    ) -> None:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()

        config = DiarizerConfig(hf_token="hf_test_token_123")
        d = PyannoteDiarizer(config)
        d.load_model()

        call_kwargs = mock_pipeline_cls.from_pretrained.call_args
        assert call_kwargs.kwargs.get("use_auth_token") == "hf_test_token_123"

    @patch("stt.core.diarizer.Pipeline")
    def test_no_hf_token_no_auth(
        self, mock_pipeline_cls: MagicMock,
    ) -> None:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()

        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        d.load_model()

        call_kwargs = mock_pipeline_cls.from_pretrained.call_args
        assert "use_auth_token" not in (call_kwargs.kwargs or {})


class TestPyannoteDiarizerErrors:
    @patch("stt.core.diarizer.Pipeline")
    def test_missing_token_raises_model_error(self, mock_pipeline_cls: MagicMock) -> None:
        mock_pipeline_cls.from_pretrained.side_effect = Exception("HF_TOKEN required")

        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        with pytest.raises(ModelError):
            d.load_model()


class TestPyannoteDiarizerLifecycle:
    @patch("stt.core.diarizer.torch")
    @patch("stt.core.diarizer.Pipeline")
    def test_load_unload(self, mock_pipeline_cls: MagicMock, mock_torch: MagicMock) -> None:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        d.load_model()
        assert d._pipeline is not None

        d.unload_model()
        assert d._pipeline is None
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("stt.core.diarizer.torch")
    @patch("stt.core.diarizer.Pipeline")
    def test_unload_calls_empty_cache_when_cuda_available(
        self, mock_pipeline_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        d.load_model()
        d.unload_model()

        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("stt.core.diarizer.torch")
    @patch("stt.core.diarizer.Pipeline")
    def test_unload_skips_empty_cache_when_no_cuda(
        self, mock_pipeline_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_pipeline_cls.from_pretrained.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        d.load_model()
        d.unload_model()

        mock_torch.cuda.empty_cache.assert_not_called()

    def test_diarize_without_load_raises(self) -> None:
        config = DiarizerConfig()
        d = PyannoteDiarizer(config)
        with pytest.raises(RuntimeError):
            d.diarize("/fake/audio.wav")
