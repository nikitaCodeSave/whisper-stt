"""Tests for stt.core.subprocess_runner."""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from stt.core.subprocess_runner import run_diarization_subprocess, run_transcription_subprocess
from stt.data_models import Segment


class TestRunTranscriptionSubprocess:
    @patch("stt.core.subprocess_runner.mp")
    def test_transcription_returns_segments(self, mock_mp: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx

        mock_queue = MagicMock()
        mock_queue.get.return_value = {
            "status": "ok",
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": None, "confidence": 0.9},
            ],
        }
        mock_ctx.Queue.return_value = mock_queue

        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_ctx.Process.return_value = mock_process

        result = run_transcription_subprocess(
            {"model_size": "large-v3", "device": "cuda", "compute_type": "float16",
             "model_dir": None, "language": "ru", "batch_size": 8, "vad_filter": True,
             "condition_on_previous_text": False, "hallucination_silence_threshold": 2.0},
            "/fake/audio.wav",
        )

        assert len(result) == 1
        assert isinstance(result[0], Segment)
        assert result[0].text == "Hello"
        mock_process.start.assert_called_once()

    @patch("stt.core.subprocess_runner.mp")
    def test_subprocess_error_raises(self, mock_mp: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx

        mock_queue = MagicMock()
        mock_queue.get.return_value = {
            "status": "error",
            "error": "CudaOomError: CUDA out of memory",
        }
        mock_ctx.Queue.return_value = mock_queue

        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_ctx.Process.return_value = mock_process

        with pytest.raises(RuntimeError, match="CudaOomError"):
            run_transcription_subprocess(
                {"model_size": "large-v3", "device": "cuda", "compute_type": "float16",
                 "model_dir": None, "language": "ru", "batch_size": 8, "vad_filter": True,
                 "condition_on_previous_text": False, "hallucination_silence_threshold": 2.0},
                "/fake/audio.wav",
            )

    @patch("stt.core.subprocess_runner.mp")
    def test_timeout_raises(self, mock_mp: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx

        mock_queue = MagicMock()
        mock_queue.get.side_effect = queue.Empty()
        mock_ctx.Queue.return_value = mock_queue

        mock_process = MagicMock()
        mock_process.is_alive.return_value = True
        mock_ctx.Process.return_value = mock_process

        with pytest.raises(RuntimeError, match="Subprocess transcription failed"):
            run_transcription_subprocess(
                {"model_size": "large-v3", "device": "cuda", "compute_type": "float16",
                 "model_dir": None, "language": "ru", "batch_size": 8, "vad_filter": True,
                 "condition_on_previous_text": False, "hallucination_silence_threshold": 2.0},
                "/fake/audio.wav",
                timeout=0.1,
            )


class TestRunDiarizationSubprocess:
    @patch("stt.core.subprocess_runner.mp")
    def test_diarization_returns_result(self, mock_mp: MagicMock) -> None:
        mock_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_ctx

        mock_queue = MagicMock()
        mock_queue.get.return_value = {
            "status": "ok",
            "turns": [
                {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            ],
            "num_speakers": 1,
        }
        mock_ctx.Queue.return_value = mock_queue

        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_ctx.Process.return_value = mock_process

        result = run_diarization_subprocess(
            {"num_speakers": None, "min_speakers": 1, "max_speakers": 8,
             "model_name": "pyannote/speaker-diarization-3.1",
             "cache_dir": None, "hf_token": None},
            "/fake/audio.wav",
        )

        assert result["num_speakers"] == 1
        assert len(result["turns"]) == 1
