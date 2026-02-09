"""Align transcription segments with diarization turns."""

from __future__ import annotations

from dataclasses import replace

from stt.core.diarizer import DiarizationResult
from stt.data_models import Segment


def _compute_overlap(
    seg_start: float, seg_end: float, turn_start: float, turn_end: float
) -> float:
    """Compute overlap duration between two time intervals."""
    overlap_start = max(seg_start, turn_start)
    overlap_end = min(seg_end, turn_end)
    return max(0.0, overlap_end - overlap_start)


def align_segments(
    segments: list[Segment], diarization: DiarizationResult
) -> list[Segment]:
    """Assign speakers to segments based on diarization turns."""
    if not segments:
        return []
    if not diarization.turns:
        return [replace(seg) for seg in segments]

    result: list[Segment] = []
    for seg in segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for turn in diarization.turns:
            overlap = _compute_overlap(seg.start, seg.end, turn.start, turn.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        if best_speaker is None:
            min_dist = float("inf")
            for turn in diarization.turns:
                dist = min(abs(seg.start - turn.end), abs(seg.end - turn.start))
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = turn.speaker
        result.append(replace(seg, speaker=best_speaker))
    return result
