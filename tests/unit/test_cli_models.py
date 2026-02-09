"""Tests for stt models subcommands — Sprint 5 CLI tests."""

from __future__ import annotations

from typer.testing import CliRunner

from stt.cli.app import app
from stt.exit_codes import ExitCode

runner = CliRunner()


class TestModelsList:
    def test_list_exits_0(self) -> None:
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0

    def test_list_shows_models(self) -> None:
        result = runner.invoke(app, ["models", "list"])
        assert "large-v3" in result.output


class TestModelsInfo:
    def test_info_known_model_exits_0(self) -> None:
        result = runner.invoke(app, ["models", "info", "large-v3"])
        assert result.exit_code == 0

    def test_info_shows_model_details(self) -> None:
        result = runner.invoke(app, ["models", "info", "large-v3"])
        assert "large-v3" in result.output

    def test_info_unknown_model_exits_4(self) -> None:
        result = runner.invoke(app, ["models", "info", "unknown-model"])
        assert result.exit_code == ExitCode.ERROR_MODEL
