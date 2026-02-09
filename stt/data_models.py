"""Data models for transcription results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(
                f"start ({self.start}) must not be greater than end ({self.end})"
            )

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptMetadata:
    source_file: str
    duration_seconds: float
    model: str = "large-v3"
    language: str = "ru"
    format_version: str = "1.0"
    diarization: bool = False
    num_speakers: int = 0
    processing_time_seconds: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class TranscriptResult:
    metadata: TranscriptMetadata
    segments: list[Segment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)
