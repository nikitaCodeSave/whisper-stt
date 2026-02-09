"""Tests for stt.core.aligner — Sprint 3 RED tests."""

from __future__ import annotations

import pytest

from stt.core.aligner import _compute_overlap, align_segments
from stt.core.diarizer import DiarizationResult, DiarizationTurn
from stt.data_models import Segment

# ---------------------------------------------------------------------------
# _compute_overlap
# ---------------------------------------------------------------------------

class TestComputeOverlap:
    def test_partial_overlap(self) -> None:
        result = _compute_overlap(0.0, 10.0, 5.0, 15.0)
        assert result == pytest.approx(5.0)

    def test_no_overlap(self) -> None:
        result = _compute_overlap(0.0, 5.0, 10.0, 15.0)
        assert result == pytest.approx(0.0)

    def test_contained_segment(self) -> None:
        result = _compute_overlap(0.0, 20.0, 5.0, 10.0)
        assert result == pytest.approx(5.0)

    def test_identical_ranges(self) -> None:
        result = _compute_overlap(5.0, 10.0, 5.0, 10.0)
        assert result == pytest.approx(5.0)

    def test_touching_boundaries(self) -> None:
        result = _compute_overlap(0.0, 5.0, 5.0, 10.0)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# align_segments
# ---------------------------------------------------------------------------

class TestAlignSegmentsSingleSpeaker:
    def test_all_segments_get_speaker_00(self) -> None:
        segments = [
            Segment(start=0.0, end=3.0, text="Hello"),
            Segment(start=3.5, end=6.0, text="World"),
        ]
        turns = [
            DiarizationTurn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        ]
        diarization = DiarizationResult(turns=turns, num_speakers=1)

        result = align_segments(segments, diarization)

        assert len(result) == 2
        for seg in result:
            assert seg.speaker == "SPEAKER_00"


class TestAlignSegmentsClearBoundaries:
    def test_segments_match_turns_exactly(self) -> None:
        segments = [
            Segment(start=0.0, end=5.0, text="Speaker zero"),
            Segment(start=5.0, end=10.0, text="Speaker one"),
        ]
        turns = [
            DiarizationTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationTurn(start=5.0, end=10.0, speaker="SPEAKER_01"),
        ]
        diarization = DiarizationResult(turns=turns, num_speakers=2)

        result = align_segments(segments, diarization)

        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"


class TestAlignSegmentsMajorityOverlap:
    def test_segment_assigned_to_majority_speaker(self) -> None:
        segments = [
            Segment(start=0.0, end=10.0, text="Overlapping"),
        ]
        turns = [
            DiarizationTurn(start=0.0, end=7.0, speaker="SPEAKER_00"),
            DiarizationTurn(start=7.0, end=15.0, speaker="SPEAKER_01"),
        ]
        diarization = DiarizationResult(turns=turns, num_speakers=2)

        result = align_segments(segments, diarization)

        # 7s overlap with SPEAKER_00 vs 3s with SPEAKER_01
        assert result[0].speaker == "SPEAKER_00"


class TestAlignSegmentsNoOverlap:
    def test_nearest_turn_fallback(self) -> None:
        segments = [
            Segment(start=20.0, end=25.0, text="After all turns"),
        ]
        turns = [
            DiarizationTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
            DiarizationTurn(start=5.0, end=10.0, speaker="SPEAKER_01"),
        ]
        diarization = DiarizationResult(turns=turns, num_speakers=2)

        result = align_segments(segments, diarization)

        # Nearest turn is SPEAKER_01 (ends at 10.0, closest to 20.0)
        assert result[0].speaker == "SPEAKER_01"


class TestAlignSegmentsPreservesFields:
    def test_text_and_confidence_preserved(self) -> None:
        segments = [
            Segment(start=0.0, end=5.0, text="Preserved text", confidence=0.95),
        ]
        turns = [
            DiarizationTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
        ]
        diarization = DiarizationResult(turns=turns, num_speakers=1)

        result = align_segments(segments, diarization)

        assert result[0].text == "Preserved text"
        assert result[0].confidence == pytest.approx(0.95)
        assert result[0].start == 0.0
        assert result[0].end == 5.0
        assert result[0].speaker == "SPEAKER_00"


class TestAlignSegmentsEdgeCases:
    def test_empty_segments(self) -> None:
        turns = [DiarizationTurn(start=0.0, end=5.0, speaker="SPEAKER_00")]
        diarization = DiarizationResult(turns=turns, num_speakers=1)

        result = align_segments([], diarization)
        assert result == []

    def test_returns_new_list(self) -> None:
        segments = [
            Segment(start=0.0, end=5.0, text="Original"),
        ]
        turns = [
            DiarizationTurn(start=0.0, end=5.0, speaker="SPEAKER_00"),
        ]
        diarization = DiarizationResult(turns=turns, num_speakers=1)

        result = align_segments(segments, diarization)

        # Original segments should be unchanged
        assert segments[0].speaker is None
        assert result[0].speaker == "SPEAKER_00"
        assert result is not segments

    def test_empty_diarization(self) -> None:
        segments = [
            Segment(start=0.0, end=5.0, text="No diarization"),
        ]
        diarization = DiarizationResult(turns=[], num_speakers=0)

        result = align_segments(segments, diarization)

        assert len(result) == 1
        assert result[0].speaker is None
