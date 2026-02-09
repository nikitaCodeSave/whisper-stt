"""Tests for stt.exporters dispatch — Sprint 4 RED tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from stt.data_models import Segment, TranscriptMetadata, TranscriptResult
from stt.exporters import export_transcript


def _make_result(source_file: str = "test.mp3") -> TranscriptResult:
    segments = [
        Segment(start=0.0, end=3.0, text="Hello world", speaker="SPEAKER_00", confidence=0.95),
    ]
    metadata = TranscriptMetadata(
        source_file=source_file,
        duration_seconds=10.0,
        model="large-v3",
        created_at=datetime(2026, 2, 9, 12, 0, 0),
    )
    return TranscriptResult(metadata=metadata, segments=segments)


class TestExportSingleFormat:
    def test_json_creates_file(self, tmp_path: Path) -> None:
        result = _make_result()
        export_transcript(result, formats="json", output_dir=tmp_path)
        assert (tmp_path / "test.json").exists()


class TestExportMultipleFormats:
    def test_three_formats(self, tmp_path: Path) -> None:
        result = _make_result()
        export_transcript(result, formats="json,txt,srt", output_dir=tmp_path)
        assert (tmp_path / "test.json").exists()
        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.srt").exists()


class TestExportStdout:
    def test_json_without_output_returns_string(self) -> None:
        result = _make_result()
        output = export_transcript(result, formats="json")
        assert isinstance(output, str)
        assert "Hello world" in output


class TestExportAutoCreateDir:
    def test_creates_nested_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        result = _make_result()
        export_transcript(result, formats="json", output_dir=nested)
        assert (nested / "test.json").exists()


class TestExportInvalidFormat:
    def test_invalid_format_raises(self) -> None:
        result = _make_result()
        with pytest.raises(ValueError):
            export_transcript(result, formats="xml")
