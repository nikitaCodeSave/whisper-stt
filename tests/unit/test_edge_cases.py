"""Tests for edge cases — Sprint 6 tests."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from stt.cli.app import app
from stt.exit_codes import ExitCode

runner = CliRunner()


class TestEmptyFile:
    def test_empty_file_exits_3(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        result = runner.invoke(app, ["transcribe", str(empty)])
        assert result.exit_code == ExitCode.ERROR_FILE


class TestGpuOom:
    def test_gpu_oom_exits_5(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        from stt.exceptions import GpuError

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)

        with patch(
            "stt.cli.transcribe.TranscriptionPipeline"
        ) as mock_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.run.side_effect = GpuError("CUDA OOM")
            mock_cls.return_value = mock_pipeline

            result = runner.invoke(app, ["transcribe", str(audio)])
            assert result.exit_code == ExitCode.ERROR_GPU
