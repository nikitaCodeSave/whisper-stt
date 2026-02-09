"""Tests for stt.core.pipeline — Sprint 5 pipeline tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stt.core.pipeline import PipelineConfig, TranscriptionPipeline
from stt.data_models import Segment, TranscriptResult


class TestPipelineConfig:
    def test_default_config(self) -> None:
        config = PipelineConfig()
        assert config.model_size == "large-v3"
        assert config.device == "cuda"
        assert config.compute_type == "float16"
        assert config.language == "ru"
        assert config.diarization_enabled is True
        assert config.num_speakers is None
        assert config.min_speakers == 1
        assert config.max_speakers == 8
        assert config.formats == "json"
        assert config.output_dir == "."
        assert config.model_dir is None

    def test_custom_config(self) -> None:
        config = PipelineConfig(
            model_size="small",
            device="cpu",
            diarization_enabled=False,
        )
        assert config.model_size == "small"
        assert config.device == "cpu"
        assert config.diarization_enabled is False


class TestPipelineModelDir:
    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.align_segments")
    @patch("stt.core.pipeline.PyannoteDiarizer")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_model_dir_passed_to_transcriber_and_diarizer(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_diarizer_cls: MagicMock,
        mock_align: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            Segment(start=0.0, end=2.0, text="Test"),
        ]
        mock_transcriber_cls.return_value = mock_transcriber

        mock_diarizer = MagicMock()
        mock_diarization_result = MagicMock()
        mock_diarization_result.num_speakers = 1
        mock_diarizer.diarize.return_value = mock_diarization_result
        mock_diarizer_cls.return_value = mock_diarizer

        mock_align.return_value = [
            Segment(start=0.0, end=2.0, text="Test", speaker="SPEAKER_00"),
        ]

        config = PipelineConfig(
            diarization_enabled=True, model_dir="/custom/models",
        )
        pipeline = TranscriptionPipeline(config)
        pipeline.run("/fake/audio.wav")

        # Verify TranscriberConfig got model_dir
        transcriber_config = mock_transcriber_cls.call_args[0][0]
        assert transcriber_config.model_dir == "/custom/models"

        # Verify DiarizerConfig got cache_dir
        diarizer_config = mock_diarizer_cls.call_args[0][0]
        assert diarizer_config.cache_dir == "/custom/models"


class TestPipelineLanguagePassthrough:
    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_language_passed_to_transcriber_config(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            Segment(start=0.0, end=2.0, text="Test"),
        ]
        mock_transcriber_cls.return_value = mock_transcriber

        config = PipelineConfig(language="en", diarization_enabled=False)
        pipeline = TranscriptionPipeline(config)
        pipeline.run("/fake/audio.wav")

        transcriber_config = mock_transcriber_cls.call_args[0][0]
        assert transcriber_config.language == "en"

    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_default_language_ru_passed(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            Segment(start=0.0, end=2.0, text="Test"),
        ]
        mock_transcriber_cls.return_value = mock_transcriber

        config = PipelineConfig(diarization_enabled=False)
        pipeline = TranscriptionPipeline(config)
        pipeline.run("/fake/audio.wav")

        transcriber_config = mock_transcriber_cls.call_args[0][0]
        assert transcriber_config.language == "ru"


class TestPipelineOutputDirOverride:
    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_output_dir_override(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            Segment(start=0.0, end=2.0, text="Test"),
        ]
        mock_transcriber_cls.return_value = mock_transcriber

        config = PipelineConfig(
            diarization_enabled=False, output_dir="/default/dir",
        )
        pipeline = TranscriptionPipeline(config)
        pipeline.run("/fake/audio.wav", output_dir="/override/dir")

        from pathlib import Path

        call_args = mock_export.call_args
        assert call_args[0][2] == Path("/override/dir")

    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_no_output_dir_override_uses_config(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            Segment(start=0.0, end=2.0, text="Test"),
        ]
        mock_transcriber_cls.return_value = mock_transcriber

        config = PipelineConfig(
            diarization_enabled=False, output_dir="/default/dir",
        )
        pipeline = TranscriptionPipeline(config)
        pipeline.run("/fake/audio.wav")

        from pathlib import Path

        call_args = mock_export.call_args
        assert call_args[0][2] == Path("/default/dir")


class TestPipelineFullRun:
    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.align_segments")
    @patch("stt.core.pipeline.PyannoteDiarizer")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_full_run_with_diarization(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_diarizer_cls: MagicMock,
        mock_align: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        # Setup transcriber mock
        mock_transcriber = MagicMock()
        segments = [
            Segment(start=0.0, end=2.0, text="Hello"),
            Segment(start=2.5, end=5.0, text="World"),
        ]
        mock_transcriber.transcribe.return_value = segments
        mock_transcriber_cls.return_value = mock_transcriber

        # Setup diarizer mock
        mock_diarizer = MagicMock()
        mock_diarization_result = MagicMock()
        mock_diarization_result.num_speakers = 2
        mock_diarizer.diarize.return_value = mock_diarization_result
        mock_diarizer_cls.return_value = mock_diarizer

        # Setup aligner mock
        aligned = [
            Segment(start=0.0, end=2.0, text="Hello", speaker="SPEAKER_00"),
            Segment(start=2.5, end=5.0, text="World", speaker="SPEAKER_01"),
        ]
        mock_align.return_value = aligned

        config = PipelineConfig(diarization_enabled=True)
        pipeline = TranscriptionPipeline(config)
        result = pipeline.run("/fake/audio.wav")

        # Verify audio validation
        mock_validate.assert_called_once()

        # Verify transcriber lifecycle
        mock_transcriber.load_model.assert_called_once()
        mock_transcriber.transcribe.assert_called_once()
        mock_transcriber.unload_model.assert_called_once()

        # Verify diarizer lifecycle
        mock_diarizer.load_model.assert_called_once()
        mock_diarizer.diarize.assert_called_once()
        mock_diarizer.unload_model.assert_called_once()

        # Verify alignment
        mock_align.assert_called_once_with(segments, mock_diarization_result)

        # Verify result
        assert isinstance(result, TranscriptResult)
        assert result.segments == aligned
        assert result.metadata.diarization is True
        assert result.metadata.num_speakers == 2

    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.align_segments")
    @patch("stt.core.pipeline.PyannoteDiarizer")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_sequential_vram_management(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_diarizer_cls: MagicMock,
        mock_align: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        """Verify unload is called BEFORE next load for VRAM."""
        call_order: list[str] = []

        mock_transcriber = MagicMock()
        mock_transcriber.load_model.side_effect = (
            lambda: call_order.append("transcriber_load")
        )
        mock_transcriber.transcribe.side_effect = (
            lambda p: call_order.append("transcribe") or []
        )
        mock_transcriber.unload_model.side_effect = (
            lambda: call_order.append("transcriber_unload")
        )
        mock_transcriber_cls.return_value = mock_transcriber

        mock_diarizer = MagicMock()
        mock_diarizer.load_model.side_effect = (
            lambda: call_order.append("diarizer_load")
        )
        mock_diarization = MagicMock()
        mock_diarization.num_speakers = 1
        mock_diarizer.diarize.side_effect = (
            lambda p: call_order.append("diarize") or mock_diarization
        )
        mock_diarizer.unload_model.side_effect = (
            lambda: call_order.append("diarizer_unload")
        )
        mock_diarizer_cls.return_value = mock_diarizer

        mock_align.return_value = []

        config = PipelineConfig(diarization_enabled=True)
        pipeline = TranscriptionPipeline(config)
        pipeline.run("/fake/audio.wav")

        # Verify sequential order: transcriber unloaded before diarizer loaded
        assert call_order.index("transcriber_load") < call_order.index(
            "transcribe"
        )
        assert call_order.index("transcribe") < call_order.index(
            "transcriber_unload"
        )
        assert call_order.index("transcriber_unload") < call_order.index(
            "diarizer_load"
        )
        assert call_order.index("diarizer_load") < call_order.index(
            "diarize"
        )
        assert call_order.index("diarize") < call_order.index(
            "diarizer_unload"
        )


class TestPipelineNoDiarize:
    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.PyannoteDiarizer")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_no_diarize_skips_diarizer(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_diarizer_cls: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mock_transcriber = MagicMock()
        segments = [Segment(start=0.0, end=2.0, text="Hello")]
        mock_transcriber.transcribe.return_value = segments
        mock_transcriber_cls.return_value = mock_transcriber

        config = PipelineConfig(diarization_enabled=False)
        pipeline = TranscriptionPipeline(config)
        result = pipeline.run("/fake/audio.wav")

        # Diarizer should NOT be instantiated
        mock_diarizer_cls.assert_not_called()

        # Segments should have no speaker
        for seg in result.segments:
            assert seg.speaker is None

        assert result.metadata.diarization is False
        assert result.metadata.num_speakers == 0


class TestPipelineResult:
    @patch("stt.core.pipeline.export_transcript")
    @patch("stt.core.pipeline.Transcriber")
    @patch("stt.core.pipeline.validate_audio_file")
    def test_returns_transcript_result(
        self,
        mock_validate: MagicMock,
        mock_transcriber_cls: MagicMock,
        mock_export: MagicMock,
    ) -> None:
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = [
            Segment(start=0.0, end=2.0, text="Test"),
        ]
        mock_transcriber_cls.return_value = mock_transcriber

        config = PipelineConfig(diarization_enabled=False)
        pipeline = TranscriptionPipeline(config)
        result = pipeline.run("/fake/audio.wav")

        assert isinstance(result, TranscriptResult)
        assert result.metadata.source_file == "/fake/audio.wav"
        assert result.metadata.model == "large-v3"
        assert result.metadata.language == "ru"
