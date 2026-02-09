"""Tests for stt.exporters.srt_export — Sprint 4 RED tests."""

from __future__ import annotations

from io import StringIO

from stt.data_models import Segment, TranscriptMetadata, TranscriptResult
from stt.exporters.srt_export import export_srt


def _make_result(segments: list[Segment]) -> TranscriptResult:
    from datetime import datetime

    metadata = TranscriptMetadata(
        source_file="test.mp3",
        duration_seconds=60.0,
        model="large-v3",
        created_at=datetime(2026, 2, 9, 12, 0, 0),
    )
    return TranscriptResult(metadata=metadata, segments=segments)


class TestSrtTimestampFormat:
    def test_timestamp_with_millis(self) -> None:
        segments = [Segment(start=83.456, end=90.123, text="Test")]
        result = _make_result(segments)
        output = StringIO()
        export_srt(result, output)
        content = output.getvalue()
        assert "00:01:23,456" in content
        assert "00:01:30,123" in content


class TestSrtNumbering:
    def test_starts_at_one(self) -> None:
        segments = [
            Segment(start=0.0, end=3.0, text="First"),
            Segment(start=3.5, end=6.0, text="Second"),
        ]
        result = _make_result(segments)
        output = StringIO()
        export_srt(result, output)
        lines = output.getvalue().strip().split("\n")
        assert lines[0] == "1"


class TestSrtArrowFormat:
    def test_arrow_with_spaces(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="Test")]
        result = _make_result(segments)
        output = StringIO()
        export_srt(result, output)
        content = output.getvalue()
        assert " --> " in content


class TestSrtSpeakerPrefix:
    def test_with_speaker(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="Hello", speaker="SPEAKER_00")]
        result = _make_result(segments)
        output = StringIO()
        export_srt(result, output)
        content = output.getvalue()
        assert "[SPEAKER_00] Hello" in content

    def test_without_speaker(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="Hello")]
        result = _make_result(segments)
        output = StringIO()
        export_srt(result, output)
        content = output.getvalue()
        # No brackets when no speaker
        assert "[" not in content
        assert "Hello" in content


class TestSrtEmptyLines:
    def test_empty_line_between_entries(self) -> None:
        segments = [
            Segment(start=0.0, end=3.0, text="First"),
            Segment(start=3.5, end=6.0, text="Second"),
        ]
        result = _make_result(segments)
        output = StringIO()
        export_srt(result, output)
        content = output.getvalue()
        # SRT entries separated by empty line
        assert "\n\n" in content
