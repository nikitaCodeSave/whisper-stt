"""Tests for stt.cli.app — Sprint 5 CLI app tests."""

from __future__ import annotations

from typer.testing import CliRunner

from stt import __version__
from stt.cli.app import app

runner = CliRunner()


class TestCliHelp:
    def test_help_shows_transcribe_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output.lower()

    def test_help_shows_models_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "models" in result.output.lower()

    def test_help_shows_batch_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "batch" in result.output.lower()


class TestCliVersion:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"stt {__version__}" in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestTranscribeHelp:
    def test_transcribe_help_shows_options(self) -> None:
        result = runner.invoke(app, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "audio" in result.output.lower()

    def test_transcribe_help_shows_model_option(self) -> None:
        result = runner.invoke(app, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_transcribe_help_shows_no_diarize(self) -> None:
        result = runner.invoke(app, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "--no-diarize" in result.output
