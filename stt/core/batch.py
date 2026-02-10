"""Batch processing for audio files."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from stt.core.audio import SUPPORTED_EXTENSIONS
from stt.core.pipeline import PipelineConfig, TranscriptionPipeline
from stt.exit_codes import ExitCode

logger = logging.getLogger(__name__)


def discover_audio_files(
    input_dir: Path,
    recursive: bool = False,
    pattern: str = "*",
) -> list[Path]:
    """Discover audio files in a directory."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {input_dir}"
        )

    if pattern != "*":
        if recursive:
            candidates = list(input_dir.rglob(pattern))
        else:
            candidates = list(input_dir.glob(pattern))
    elif recursive:
        candidates = list(input_dir.rglob("*"))
    else:
        candidates = list(input_dir.iterdir())

    return sorted(
        f
        for f in candidates
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


@dataclass
class BatchResult:
    total: int
    succeeded: int
    failed: int
    errors: list[tuple[Path, str]]

    @property
    def exit_code(self) -> ExitCode:
        if self.failed == 0:
            return ExitCode.SUCCESS
        if self.succeeded > 0:
            return ExitCode.PARTIAL_SUCCESS
        return ExitCode.ERROR_GENERAL


class BatchRunner:
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        skip_existing: bool = False,
    ) -> None:
        self._config = pipeline_config
        self._skip_existing = skip_existing

    def run(
        self,
        files: list[Path],
        output_dir: Path,
        input_base: Path | None = None,
    ) -> BatchResult:
        succeeded = 0
        failed = 0
        errors: list[tuple[Path, str]] = []

        pipeline = TranscriptionPipeline(self._config)

        for audio_file in files:
            # Determine output subdirectory
            if input_base is not None:
                try:
                    rel = audio_file.parent.relative_to(input_base)
                    file_output_dir = output_dir / rel
                except ValueError:
                    file_output_dir = output_dir
            else:
                file_output_dir = output_dir

            # Check skip-existing
            if self._skip_existing:
                fmt_list = [
                    f.strip()
                    for f in self._config.formats.split(",")
                ]
                stem = audio_file.stem
                all_exist = all(
                    (file_output_dir / f"{stem}.{fmt}").exists()
                    for fmt in fmt_list
                )
                if all_exist:
                    succeeded += 1
                    continue

            # Run pipeline with per-file output_dir
            needs_cleanup = False
            try:
                pipeline.run(
                    str(audio_file),
                    output_dir=str(file_output_dir),
                )
                succeeded += 1
            except Exception as e:
                failed += 1
                errors.append((audio_file, str(e)))
                logger.error("Failed %s: %s", audio_file, e, exc_info=True)
                needs_cleanup = True
            if needs_cleanup:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return BatchResult(
            total=len(files),
            succeeded=succeeded,
            failed=failed,
            errors=errors,
        )
