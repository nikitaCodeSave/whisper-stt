"""Audio file validation."""

from __future__ import annotations

from pathlib import Path

from stt.exceptions import AudioValidationError

SUPPORTED_EXTENSIONS: set[str] = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}


def validate_audio_file(path: Path) -> None:
    if not path.exists():
        raise AudioValidationError(f"File not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if path.stat().st_size == 0:
        raise AudioValidationError(f"File is empty: {path}")
