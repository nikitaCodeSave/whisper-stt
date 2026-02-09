"""Audio file validation and preprocessing."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stt.exceptions import AudioPreprocessError, AudioValidationError

logger = logging.getLogger(__name__)

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


@dataclass
class PreprocessedAudio:
    """Holds the path to a preprocessed WAV file and handles cleanup."""

    path: Path

    def cleanup(self) -> None:
        """Remove the temporary preprocessed file. Safe to call multiple times."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove temp file: %s", self.path)


_FFMPEG_TIMEOUT = 300


def preprocess_audio(source: Path) -> PreprocessedAudio:
    """Convert audio to WAV 16kHz mono PCM_S16LE via ffmpeg.

    Always converts regardless of source format to guarantee a consistent
    input for both faster-whisper and pyannote.
    """
    fd, tmp_path_str = tempfile.mkstemp(suffix=".wav", dir=source.parent)
    # Close the fd immediately — ffmpeg will write to the path directly.
    import os

    os.close(fd)
    tmp_path = Path(tmp_path_str)

    cmd = [
        "ffmpeg",
        "-i", str(source),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-y",
        str(tmp_path),
    ]

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT,
            check=False,
        )
    except FileNotFoundError:
        tmp_path.unlink(missing_ok=True)
        raise AudioPreprocessError(
            "ffmpeg not found. Install ffmpeg to process audio files."
        ) from None
    except subprocess.TimeoutExpired:
        tmp_path.unlink(missing_ok=True)
        raise AudioPreprocessError(
            f"ffmpeg timed out after {_FFMPEG_TIMEOUT}s while converting {source.name}"
        ) from None
    else:
        # Check for non-zero exit or missing/empty output
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            raise AudioPreprocessError(
                f"ffmpeg failed to convert {source.name} to WAV 16kHz mono."
            )

    return PreprocessedAudio(path=tmp_path)
