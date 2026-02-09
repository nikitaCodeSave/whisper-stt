"""TXT exporter for transcription results."""

from __future__ import annotations

from typing import IO

from stt.data_models import TranscriptResult


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def export_txt(result: TranscriptResult, output: IO[str]) -> None:
    """Write transcription result as plain text to the given output stream."""
    lines = []
    for seg in result.segments:
        ts = _format_timestamp(seg.start)
        if seg.speaker is not None:
            lines.append(f"[{ts}] {seg.speaker}: {seg.text}")
        else:
            lines.append(f"[{ts}] {seg.text}")
    output.write("\n".join(lines) + "\n")
