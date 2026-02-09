"""Tests for stt models command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from stt.cli.app import app
from stt.exit_codes import ExitCode

runner = CliRunner()


class TestModelsDownloadInvalidModel:
    def test_unknown_model_exits_4(self) -> None:
        result = runner.invoke(
            app, ["models", "download", "nonexistent-model"],
        )
        assert result.exit_code == ExitCode.ERROR_MODEL


class TestModelsDownloadValid:
    @patch("stt.cli.models_cmd.load_config")
    def test_valid_model_triggers_whisper_download(
        self, mock_load_config: MagicMock,
    ) -> None:
        from stt.config import SttConfig

        mock_load_config.return_value = SttConfig(
            model_dir="/tmp/test-models",
        )

        with patch(
            "faster_whisper.WhisperModel",
        ) as mock_whisper:
            result = runner.invoke(
                app, ["models", "download", "tiny"],
            )

        assert result.exit_code == 0
        mock_whisper.assert_called_once()
        call_kwargs = mock_whisper.call_args
        assert call_kwargs[0][0] == "tiny"


class TestModelsDownloadModelDir:
    def test_model_dir_option_accepted(self) -> None:
        result = runner.invoke(
            app,
            [
                "models", "download", "nonexistent",
                "--model-dir", "/custom/path",
            ],
        )
        assert result.exit_code == ExitCode.ERROR_MODEL


class TestModelsList:
    def test_list_shows_models(self) -> None:
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "tiny" in result.output
        assert "large-v3" in result.output


class TestModelsInfo:
    def test_info_known_model(self) -> None:
        result = runner.invoke(app, ["models", "info", "tiny"])
        assert result.exit_code == 0
        assert "tiny" in result.output

    def test_info_unknown_model(self) -> None:
        result = runner.invoke(app, ["models", "info", "nonexistent"])
        assert result.exit_code == ExitCode.ERROR_MODEL
