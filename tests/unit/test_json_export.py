"""Tests for stt.exporters.json_export — Sprint 4 RED tests."""

from __future__ import annotations

import json
from datetime import datetime
from io import StringIO
from pathlib import Path

from stt.data_models import Segment, TranscriptMetadata, TranscriptResult
from stt.exporters.json_export import export_json


def _make_result(
    segments: list[Segment] | None = None,
    source_file: str = "test.mp3",
    duration: float = 10.0,
) -> TranscriptResult:
    if segments is None:
        segments = [
            Segment(start=0.0, end=3.0, text="Hello world"),
            Segment(
                start=3.5, end=6.0, text="Second segment",
                speaker="SPEAKER_00", confidence=0.92,
            ),
        ]
    metadata = TranscriptMetadata(
        source_file=source_file,
        duration_seconds=duration,
        model="large-v3",
        created_at=datetime(2026, 2, 9, 12, 0, 0),
    )
    return TranscriptResult(metadata=metadata, segments=segments)


class TestJsonExportStructure:
    def test_has_required_top_level_keys(self) -> None:
        result = _make_result()
        output = StringIO()
        export_json(result, output)
        data = json.loads(output.getvalue())
        assert "metadata" in data
        assert "segments" in data
        assert "full_text" in data

    def test_metadata_fields(self) -> None:
        result = _make_result()
        output = StringIO()
        export_json(result, output)
        data = json.loads(output.getvalue())
        meta = data["metadata"]
        assert "format_version" in meta
        assert "source_file" in meta
        assert "duration_seconds" in meta
        assert "model" in meta
        assert "created_at" in meta

    def test_segment_fields(self) -> None:
        result = _make_result()
        output = StringIO()
        export_json(result, output)
        data = json.loads(output.getvalue())
        seg = data["segments"][1]
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert "speaker" in seg
        assert "confidence" in seg


class TestJsonExportOptionalFields:
    def test_speaker_none_not_in_output(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="No speaker")]
        result = _make_result(segments=segments)
        output = StringIO()
        export_json(result, output)
        data = json.loads(output.getvalue())
        seg = data["segments"][0]
        assert "speaker" not in seg

    def test_confidence_none_not_in_output(self) -> None:
        segments = [Segment(start=0.0, end=3.0, text="No confidence")]
        result = _make_result(segments=segments)
        output = StringIO()
        export_json(result, output)
        data = json.loads(output.getvalue())
        seg = data["segments"][0]
        assert "confidence" not in seg


class TestJsonExportFile:
    def test_write_to_file(self, tmp_path: Path) -> None:
        result = _make_result()
        out_file = tmp_path / "output.json"
        with open(out_file, "w") as f:
            export_json(result, f)
        data = json.loads(out_file.read_text())
        assert data["metadata"]["source_file"] == "test.mp3"


class TestJsonExportFullText:
    def test_full_text_matches_segments(self) -> None:
        segments = [
            Segment(start=0.0, end=2.0, text="Hello"),
            Segment(start=2.5, end=5.0, text="World"),
        ]
        result = _make_result(segments=segments)
        output = StringIO()
        export_json(result, output)
        data = json.loads(output.getvalue())
        assert data["full_text"] == "Hello World"
