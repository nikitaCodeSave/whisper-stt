"""Tests for stt transcribe command — Sprint 5 CLI tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from stt.cli.app import app
from stt.exit_codes import ExitCode

runner = CliRunner()


class TestTranscribeMissingFile:
    def test_missing_file_exits_3(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["transcribe", str(tmp_path / "nonexistent.wav")]
        )
        assert result.exit_code == ExitCode.ERROR_FILE


class TestTranscribeBadExtension:
    def test_txt_extension_exits_3(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("not audio")
        result = runner.invoke(app, ["transcribe", str(bad_file)])
        assert result.exit_code == ExitCode.ERROR_FILE


class TestTranscribeSpeakerConflict:
    def test_num_speakers_with_min_speakers_exits_2(
        self, tmp_path: Path,
    ) -> None:
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)
        result = runner.invoke(
            app,
            [
                "transcribe",
                str(audio),
                "--num-speakers", "2",
                "--min-speakers", "3",
            ],
        )
        assert result.exit_code == ExitCode.ERROR_ARGS

    def test_num_speakers_with_max_speakers_exits_2(
        self, tmp_path: Path,
    ) -> None:
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)
        result = runner.invoke(
            app,
            [
                "transcribe",
                str(audio),
                "--num-speakers", "2",
                "--max-speakers", "5",
            ],
        )
        assert result.exit_code == ExitCode.ERROR_ARGS


class TestTranscribeYamlCascade:
    @patch("stt.cli.transcribe.TranscriptionPipeline")
    @patch("stt.cli.transcribe.load_config")
    def test_yaml_config_used_when_cli_not_set(
        self, mock_load_config: MagicMock,
        mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        from stt.config import SttConfig

        mock_load_config.return_value = SttConfig(
            model="small",
            language="en",
            format="txt",
            device="cpu",
            compute_type="int8",
        )

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(
            app, ["transcribe", str(audio), "--no-diarize"],
        )
        assert result.exit_code == 0

        config = mock_pipeline_cls.call_args[0][0]
        assert config.model_size == "small"
        assert config.language == "en"
        assert config.formats == "txt"
        assert config.device == "cpu"
        assert config.compute_type == "int8"

    @patch("stt.cli.transcribe.TranscriptionPipeline")
    @patch("stt.cli.transcribe.load_config")
    def test_cli_overrides_yaml(
        self, mock_load_config: MagicMock,
        mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        from stt.config import SttConfig

        mock_load_config.return_value = SttConfig(
            model="small",
            language="en",
        )

        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(
            app,
            [
                "transcribe", str(audio),
                "--model", "medium",
                "--language", "de",
                "--no-diarize",
            ],
        )
        assert result.exit_code == 0

        config = mock_pipeline_cls.call_args[0][0]
        assert config.model_size == "medium"
        assert config.language == "de"


class TestTranscribeNoDiarize:
    @patch("stt.cli.transcribe.TranscriptionPipeline")
    def test_no_diarize_flag_accepted(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"\x00" * 100)

        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_pipeline.run.return_value = mock_result
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(
            app,
            ["transcribe", str(audio), "--no-diarize"],
        )
        assert result.exit_code == 0
        # Verify pipeline was created with diarization disabled
        config = mock_pipeline_cls.call_args[0][0]
        assert config.diarization_enabled is False
