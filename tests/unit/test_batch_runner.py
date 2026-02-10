"""Tests for stt.core.batch BatchRunner — Sprint 6 tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from stt.core.batch import BatchResult, BatchRunner
from stt.core.pipeline import PipelineConfig
from stt.data_models import Segment, TranscriptMetadata, TranscriptResult
from stt.exit_codes import ExitCode


def _make_result(source: str) -> TranscriptResult:
    return TranscriptResult(
        metadata=TranscriptMetadata(
            source_file=source,
            duration_seconds=10.0,
        ),
        segments=[Segment(start=0.0, end=2.0, text="test")],
    )


class TestBatchResult:
    def test_all_success_exit_code_0(self) -> None:
        br = BatchResult(total=3, succeeded=3, failed=0, errors=[])
        assert br.exit_code == ExitCode.SUCCESS

    def test_partial_fail_exit_code_10(self) -> None:
        br = BatchResult(
            total=3,
            succeeded=2,
            failed=1,
            errors=[(Path("a.mp3"), "err")],
        )
        assert br.exit_code == ExitCode.PARTIAL_SUCCESS

    def test_all_fail_exit_code_1(self) -> None:
        br = BatchResult(
            total=2,
            succeeded=0,
            failed=2,
            errors=[
                (Path("a.mp3"), "err1"),
                (Path("b.mp3"), "err2"),
            ],
        )
        assert br.exit_code == ExitCode.ERROR_GENERAL


class TestBatchRunnerSuccess:
    @patch("stt.core.batch.TranscriptionPipeline")
    def test_all_success(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        files = [tmp_path / "a.mp3", tmp_path / "b.wav"]
        for f in files:
            f.write_bytes(b"\x00" * 10)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = [
            _make_result(str(files[0])),
            _make_result(str(files[1])),
        ]
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig()
        runner = BatchRunner(config, skip_existing=False)
        result = runner.run(files, tmp_path / "output")

        assert result.total == 2
        assert result.succeeded == 2
        assert result.failed == 0
        assert result.exit_code == ExitCode.SUCCESS


class TestBatchRunnerPartialFail:
    @patch("stt.core.batch.TranscriptionPipeline")
    def test_partial_failure(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        files = [tmp_path / "a.mp3", tmp_path / "b.wav"]
        for f in files:
            f.write_bytes(b"\x00" * 10)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = [
            _make_result(str(files[0])),
            Exception("processing error"),
        ]
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig()
        runner = BatchRunner(config, skip_existing=False)
        result = runner.run(files, tmp_path / "output")

        assert result.total == 2
        assert result.succeeded == 1
        assert result.failed == 1
        assert result.exit_code == ExitCode.PARTIAL_SUCCESS


class TestBatchRunnerAllFail:
    @patch("stt.core.batch.TranscriptionPipeline")
    def test_all_fail(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        files = [tmp_path / "a.mp3", tmp_path / "b.wav"]
        for f in files:
            f.write_bytes(b"\x00" * 10)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = [
            Exception("err1"),
            Exception("err2"),
        ]
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig()
        runner = BatchRunner(config, skip_existing=False)
        result = runner.run(files, tmp_path / "output")

        assert result.total == 2
        assert result.succeeded == 0
        assert result.failed == 2
        assert result.exit_code == ExitCode.ERROR_GENERAL


class TestBatchRunnerSkipExisting:
    @patch("stt.core.batch.TranscriptionPipeline")
    def test_skip_existing_skips_processed(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"\x00" * 10)

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "a.json").write_text("{}")

        mock_pipeline = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig(formats="json")
        runner = BatchRunner(config, skip_existing=True)
        result = runner.run([audio], output_dir)

        # File was skipped, so pipeline.run should NOT be called
        mock_pipeline.run.assert_not_called()
        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0


class TestBatchRunnerOutputStructure:
    @patch("stt.core.batch.TranscriptionPipeline")
    def test_output_dir_preserves_structure(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        sub = tmp_path / "input" / "subdir"
        sub.mkdir(parents=True)
        audio = sub / "test.mp3"
        audio.write_bytes(b"\x00" * 10)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = _make_result(str(audio))
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig()
        runner = BatchRunner(config, skip_existing=False)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        result = runner.run(
            [audio], output_dir, input_base=input_dir,
        )

        assert result.succeeded == 1


class TestBatchRunnerPipelineReuse:
    @patch("stt.core.batch.TranscriptionPipeline")
    def test_pipeline_created_once_for_multiple_files(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        files = [tmp_path / "a.mp3", tmp_path / "b.wav", tmp_path / "c.flac"]
        for f in files:
            f.write_bytes(b"\x00" * 10)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = _make_result("test")
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig()
        runner = BatchRunner(config, skip_existing=False)
        result = runner.run(files, tmp_path / "output")

        # Pipeline constructor called exactly once
        mock_pipeline_cls.assert_called_once()
        # But run called for each file
        assert mock_pipeline.run.call_count == 3
        assert result.succeeded == 3

    @patch("stt.core.batch.TranscriptionPipeline")
    def test_pipeline_run_receives_output_dir(
        self, mock_pipeline_cls: MagicMock, tmp_path: Path,
    ) -> None:
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00" * 10)

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = _make_result(str(audio))
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig()
        runner = BatchRunner(config, skip_existing=False)
        output_dir = tmp_path / "output"
        runner.run([audio], output_dir)

        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs.kwargs.get("output_dir") == str(output_dir)


class TestBatchRunnerVramCleanup:
    @patch("stt.core.batch.torch")
    @patch("stt.core.batch.gc")
    @patch("stt.core.batch.TranscriptionPipeline")
    def test_vram_cleanup_on_failure(
        self,
        mock_pipeline_cls: MagicMock,
        mock_gc: MagicMock,
        mock_torch: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        files = [tmp_path / "a.mp3"]
        files[0].write_bytes(b"\x00" * 10)

        mock_pipeline = MagicMock()
        mock_pipeline.run.side_effect = Exception("crash")
        mock_pipeline_cls.return_value = mock_pipeline

        config = PipelineConfig()
        runner_obj = BatchRunner(config, skip_existing=False)
        result = runner_obj.run(files, tmp_path / "output")

        assert result.failed == 1
        mock_gc.collect.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()
