"""Tests for audio preprocessing (ffmpeg conversion)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from stt.core.audio import preprocess_audio
from stt.exceptions import AudioPreprocessError


@pytest.fixture()
def minimal_wav(tmp_path: Path) -> Path:
    """Create a minimal valid WAV file (silence, 16kHz mono, 0.1s)."""
    import struct
    import wave

    wav_path = tmp_path / "test.wav"
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # 0.1s of silence = 1600 frames
        wf.writeframes(struct.pack("<" + "h" * 1600, *([0] * 1600)))
    return wav_path


class TestPreprocessAudio:
    def test_preprocess_creates_wav_file(self, minimal_wav: Path) -> None:
        result = preprocess_audio(minimal_wav)
        try:
            assert result.path.exists()
            assert result.path.suffix == ".wav"
            assert result.path != minimal_wav
            assert result.path.stat().st_size > 0
        finally:
            result.cleanup()

    def test_cleanup_removes_temp_file(self, minimal_wav: Path) -> None:
        result = preprocess_audio(minimal_wav)
        temp_path = result.path
        assert temp_path.exists()
        result.cleanup()
        assert not temp_path.exists()

    def test_cleanup_idempotent(self, minimal_wav: Path) -> None:
        result = preprocess_audio(minimal_wav)
        result.cleanup()
        # Second call should not raise
        result.cleanup()

    @patch("stt.core.audio.subprocess.run")
    def test_ffmpeg_not_found_raises_error(
        self, mock_run: patch, minimal_wav: Path,
    ) -> None:
        mock_run.side_effect = FileNotFoundError("ffmpeg")
        with pytest.raises(AudioPreprocessError, match="ffmpeg not found"):
            preprocess_audio(minimal_wav)

    @patch("stt.core.audio.subprocess.run")
    def test_ffmpeg_failure_raises_error(
        self, mock_run: patch, minimal_wav: Path,
    ) -> None:
        # Simulate ffmpeg running but returning non-zero exit code
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"error",
        )
        with pytest.raises(AudioPreprocessError, match="ffmpeg failed"):
            preprocess_audio(minimal_wav)

    @patch("stt.core.audio.subprocess.run")
    def test_ffmpeg_timeout_raises_error(
        self, mock_run: patch, minimal_wav: Path,
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)
        with pytest.raises(AudioPreprocessError, match="timed out"):
            preprocess_audio(minimal_wav)

    @patch("stt.core.audio.subprocess.run")
    def test_temp_file_cleaned_on_failure(
        self, mock_run: patch, minimal_wav: Path,
    ) -> None:
        mock_run.side_effect = FileNotFoundError("ffmpeg")
        with pytest.raises(AudioPreprocessError):
            preprocess_audio(minimal_wav)
        # No orphan temp files in the source directory
        temp_files = list(minimal_wav.parent.glob("*.wav"))
        # Only the original minimal_wav should remain
        assert temp_files == [minimal_wav]


class TestPreprocessFfmpegReturncode:
    @patch("stt.core.audio.subprocess.run")
    def test_ffmpeg_nonzero_returncode_raises(
        self, mock_run: patch, minimal_wav: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=b"", stderr=b"error converting file",
        )
        with pytest.raises(AudioPreprocessError, match="ffmpeg failed \\(code 1\\)"):
            preprocess_audio(minimal_wav)

    @patch("stt.core.audio.subprocess.run")
    def test_ffmpeg_error_includes_stderr(
        self, mock_run: patch, minimal_wav: Path,
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=2, stdout=b"",
            stderr=b"Invalid data found when processing input",
        )
        with pytest.raises(AudioPreprocessError, match="Invalid data found"):
            preprocess_audio(minimal_wav)
