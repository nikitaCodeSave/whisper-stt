"""Speaker hints for diarization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeakerHints:
    num_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 8


def validate_speaker_hints(hints: SpeakerHints) -> SpeakerHints:
    if hints.num_speakers is not None:
        return SpeakerHints(
            num_speakers=hints.num_speakers,
            min_speakers=hints.num_speakers,
            max_speakers=hints.num_speakers,
        )

    if hints.min_speakers > hints.max_speakers:
        raise ValueError(
            f"min_speakers ({hints.min_speakers}) must not exceed "
            f"max_speakers ({hints.max_speakers})"
        )

    return hints
