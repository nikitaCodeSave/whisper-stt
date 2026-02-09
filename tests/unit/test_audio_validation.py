"""Sprint 1: Tests for audio file validation."""

import wave
from pathlib import Path

import pytest

from stt.core.audio import SUPPORTED_EXTENSIONS, validate_audio_file
from stt.exceptions import AudioValidationError


def _create_minimal_wav(path: Path) -> None:
    """Create a minimal valid WAV file for testing."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # Write 1600 frames (0.1 seconds of silence)
        wf.writeframes(b"\x00\x00" * 1600)


class TestValidateAudioFileNonexistent:
    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(AudioValidationError):
            validate_audio_file(Path("/tmp/nonexistent_audio_file_12345.mp3"))


class TestValidateAudioFileWrongExtension:
    def test_txt_extension_raises(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("not audio")
        with pytest.raises(AudioValidationError):
            validate_audio_file(txt_file)

    def test_py_extension_raises(self, tmp_path: Path) -> None:
        py_file = tmp_path / "script.py"
        py_file.write_text("print('hello')")
        with pytest.raises(AudioValidationError):
            validate_audio_file(py_file)


class TestValidateAudioFileEmpty:
    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty_wav = tmp_path / "empty.wav"
        empty_wav.write_bytes(b"")
        with pytest.raises(AudioValidationError):
            validate_audio_file(empty_wav)


class TestValidateAudioFileValid:
    def test_valid_wav_passes(self, tmp_path: Path) -> None:
        wav_file = tmp_path / "test.wav"
        _create_minimal_wav(wav_file)
        # Should not raise
        validate_audio_file(wav_file)


class TestSupportedExtensions:
    def test_contains_mp3(self) -> None:
        assert ".mp3" in SUPPORTED_EXTENSIONS

    def test_contains_wav(self) -> None:
        assert ".wav" in SUPPORTED_EXTENSIONS

    def test_contains_flac(self) -> None:
        assert ".flac" in SUPPORTED_EXTENSIONS

    def test_contains_m4a(self) -> None:
        assert ".m4a" in SUPPORTED_EXTENSIONS

    def test_contains_ogg(self) -> None:
        assert ".ogg" in SUPPORTED_EXTENSIONS

    def test_contains_opus(self) -> None:
        assert ".opus" in SUPPORTED_EXTENSIONS

    def test_is_complete_set(self) -> None:
        expected = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}
        assert SUPPORTED_EXTENSIONS == expected
