"""Tests for stt.exporters.txt_export — Sprint 4 RED tests."""

from __future__ import annotations

from io import StringIO

from stt.data_models import Segment, TranscriptMetadata, TranscriptResult
from stt.exporters.txt_export import export_txt


def _make_result(segments: list[Segment]) -> TranscriptResult:
    from datetime import datetime

    metadata = TranscriptMetadata(
        source_file="test.mp3",
        duration_seconds=60.0,
        model="large-v3",
        created_at=datetime(2026, 2, 9, 12, 0, 0),
    )
    return TranscriptResult(metadata=metadata, segments=segments)


class TestTxtTimestampFormat:
    def test_hh_mm_ss_format(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="Hello")]
        result = _make_result(segments)
        output = StringIO()
        export_txt(result, output)
        line = output.getvalue().strip()
        assert line.startswith("[00:00:00]")

    def test_seconds_to_hhmmss_conversion(self) -> None:
        segments = [Segment(start=3661.0, end=3665.0, text="Late")]
        result = _make_result(segments)
        output = StringIO()
        export_txt(result, output)
        line = output.getvalue().strip()
        assert "[01:01:01]" in line


class TestTxtWithSpeaker:
    def test_speaker_format(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="Hello", speaker="SPEAKER_00")]
        result = _make_result(segments)
        output = StringIO()
        export_txt(result, output)
        line = output.getvalue().strip()
        assert line == "[00:00:00] SPEAKER_00: Hello"


class TestTxtWithoutSpeaker:
    def test_no_speaker_format(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="Hello")]
        result = _make_result(segments)
        output = StringIO()
        export_txt(result, output)
        line = output.getvalue().strip()
        assert line == "[00:00:00] Hello"


class TestTxtUtf8:
    def test_cyrillic_preserved(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="Привет мир")]
        result = _make_result(segments)
        output = StringIO()
        export_txt(result, output)
        content = output.getvalue()
        assert "Привет мир" in content
