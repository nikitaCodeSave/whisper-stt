"""Tests for stt batch command — Sprint 6 CLI tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from stt.cli.app import app
from stt.exit_codes import ExitCode

runner = CliRunner()


class TestBatchBasic:
    @patch("stt.cli.batch.BatchRunner")
    @patch("stt.cli.batch.discover_audio_files")
    def test_batch_command_works(
        self,
        mock_discover: MagicMock,
        mock_runner_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        audio = input_dir / "test.mp3"
        audio.write_bytes(b"\x00" * 10)

        mock_discover.return_value = [audio]

        mock_batch_result = MagicMock()
        mock_batch_result.exit_code = ExitCode.SUCCESS
        mock_batch_result.total = 1
        mock_batch_result.succeeded = 1
        mock_batch_result.failed = 0
        mock_batch_result.errors = []

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_batch_result
        mock_runner_cls.return_value = mock_runner

        result = runner.invoke(
            app,
            ["batch", str(input_dir), "--output", str(tmp_path / "out")],
        )
        assert result.exit_code == 0


class TestBatchNonexistentDir:
    def test_nonexistent_dir_exits_3(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["batch", str(tmp_path / "nonexistent")]
        )
        assert result.exit_code == ExitCode.ERROR_FILE


class TestBatchEmptyDir:
    @patch("stt.cli.batch.discover_audio_files")
    def test_empty_dir_shows_warning(
        self, mock_discover: MagicMock, tmp_path: Path,
    ) -> None:
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        mock_discover.return_value = []

        result = runner.invoke(app, ["batch", str(input_dir)])
        # Should warn and exit cleanly or with appropriate code
        assert "no audio" in result.output.lower() or result.exit_code == 0
