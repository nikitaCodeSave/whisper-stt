"""SRT exporter for transcription results."""

from __future__ import annotations

from typing import IO

from stt.data_models import TranscriptResult


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT."""
    total_ms = round(seconds * 1000)
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_srt(result: TranscriptResult, output: IO[str]) -> None:
    """Write transcription result as SRT subtitles to the given output stream."""
    entries = []
    for i, seg in enumerate(result.segments, start=1):
        start_ts = _format_srt_timestamp(seg.start)
        end_ts = _format_srt_timestamp(seg.end)
        if seg.speaker is not None:
            text = f"[{seg.speaker}] {seg.text}"
        else:
            text = seg.text
        entries.append(f"{i}\n{start_ts} --> {end_ts}\n{text}")
    output.write("\n\n".join(entries) + "\n")
