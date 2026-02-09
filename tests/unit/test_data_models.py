"""Sprint 1: Tests for data model dataclasses."""

from datetime import UTC, datetime

import pytest

from stt.data_models import Segment, TranscriptMetadata, TranscriptResult


class TestSegment:
    def test_create_segment_required_fields(self) -> None:
        seg = Segment(start=0.0, end=1.5, text="hello")
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.text == "hello"

    def test_speaker_defaults_to_none(self) -> None:
        seg = Segment(start=0.0, end=1.0, text="hi")
        assert seg.speaker is None

    def test_confidence_defaults_to_none(self) -> None:
        seg = Segment(start=0.0, end=1.0, text="hi")
        assert seg.confidence is None

    def test_speaker_and_confidence_set(self) -> None:
        seg = Segment(start=0.0, end=5.0, text="text", speaker="SPEAKER_00", confidence=0.95)
        assert seg.speaker == "SPEAKER_00"
        assert seg.confidence == 0.95

    def test_duration_property(self) -> None:
        seg = Segment(start=1.0, end=3.5, text="test")
        assert seg.duration == pytest.approx(2.5)

    def test_duration_property_zero_length(self) -> None:
        seg = Segment(start=5.0, end=5.0, text="")
        assert seg.duration == pytest.approx(0.0)

    def test_start_after_end_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            Segment(start=5.0, end=3.0, text="invalid")

    def test_start_equals_end_is_valid(self) -> None:
        seg = Segment(start=2.0, end=2.0, text="")
        assert seg.duration == pytest.approx(0.0)


class TestTranscriptMetadata:
    def test_create_with_required_fields(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=120.0)
        assert meta.source_file == "test.mp3"
        assert meta.duration_seconds == 120.0

    def test_default_model(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        assert meta.model == "large-v3"

    def test_default_language(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        assert meta.language == "ru"

    def test_default_format_version(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        assert meta.format_version == "1.0"

    def test_default_diarization(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        assert meta.diarization is False

    def test_default_num_speakers(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        assert meta.num_speakers == 0

    def test_default_processing_time(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        assert meta.processing_time_seconds == 0.0

    def test_created_at_auto_set(self) -> None:
        before = datetime.now(UTC)
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        after = datetime.now(UTC)
        assert isinstance(meta.created_at, datetime)
        assert before <= meta.created_at <= after

    def test_custom_values(self) -> None:
        meta = TranscriptMetadata(
            source_file="call.wav",
            duration_seconds=300.0,
            model="small",
            language="en",
            diarization=True,
            num_speakers=3,
            processing_time_seconds=45.2,
        )
        assert meta.model == "small"
        assert meta.language == "en"
        assert meta.diarization is True
        assert meta.num_speakers == 3
        assert meta.processing_time_seconds == 45.2


class TestTranscriptResult:
    def test_create_result(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        segments = [
            Segment(start=0.0, end=2.0, text="Hello"),
            Segment(start=2.5, end=5.0, text="World"),
        ]
        result = TranscriptResult(metadata=meta, segments=segments)
        assert result.metadata is meta
        assert len(result.segments) == 2

    def test_full_text_property(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        segments = [
            Segment(start=0.0, end=2.0, text="Hello"),
            Segment(start=2.5, end=5.0, text="World"),
        ]
        result = TranscriptResult(metadata=meta, segments=segments)
        assert result.full_text == "Hello World"

    def test_full_text_empty_segments(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        result = TranscriptResult(metadata=meta, segments=[])
        assert result.full_text == ""

    def test_full_text_single_segment(self) -> None:
        meta = TranscriptMetadata(source_file="test.mp3", duration_seconds=60.0)
        segments = [Segment(start=0.0, end=1.0, text="Only one")]
        result = TranscriptResult(metadata=meta, segments=segments)
        assert result.full_text == "Only one"
