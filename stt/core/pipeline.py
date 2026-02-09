"""Transcription pipeline orchestrator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from stt.core.aligner import align_segments
from stt.core.audio import validate_audio_file
from stt.core.diarizer import DiarizerConfig, PyannoteDiarizer
from stt.core.transcriber import Transcriber, TranscriberConfig
from stt.data_models import TranscriptMetadata, TranscriptResult
from stt.exporters import export_transcript


@dataclass
class PipelineConfig:
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "ru"
    diarization_enabled: bool = True
    num_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 8
    formats: str = "json"
    output_dir: str = "."
    model_dir: str | None = None
    hf_token: str | None = None


class TranscriptionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, audio_path: str, output_dir: str | None = None) -> TranscriptResult:
        start_time = time.monotonic()

        # 1. Validate audio
        validate_audio_file(Path(audio_path))

        # 2. Transcribe: load, run, unload (free VRAM)
        transcriber_config = TranscriberConfig(
            model_size=self._config.model_size,
            device=self._config.device,
            compute_type=self._config.compute_type,
            model_dir=self._config.model_dir,
            language=self._config.language,
        )
        transcriber = Transcriber(transcriber_config)
        transcriber.load_model()
        segments = transcriber.transcribe(audio_path)
        transcriber.unload_model()

        # 3. Diarize if enabled: load, run, unload (free VRAM)
        num_speakers = 0
        if self._config.diarization_enabled:
            diarizer_config = DiarizerConfig(
                num_speakers=self._config.num_speakers,
                min_speakers=self._config.min_speakers,
                max_speakers=self._config.max_speakers,
                cache_dir=self._config.model_dir,
                hf_token=self._config.hf_token,
            )
            diarizer = PyannoteDiarizer(diarizer_config)
            diarizer.load_model()
            diarization_result = diarizer.diarize(audio_path)
            diarizer.unload_model()

            # 4. Align segments with diarization
            segments = align_segments(segments, diarization_result)
            num_speakers = diarization_result.num_speakers

        # 5. Build result
        elapsed = time.monotonic() - start_time
        duration = segments[-1].end if segments else 0.0
        metadata = TranscriptMetadata(
            source_file=audio_path,
            duration_seconds=duration,
            model=self._config.model_size,
            language=self._config.language,
            diarization=self._config.diarization_enabled,
            num_speakers=num_speakers,
            processing_time_seconds=elapsed,
        )
        result = TranscriptResult(
            metadata=metadata, segments=segments,
        )

        # 6. Export
        resolved_dir = output_dir if output_dir is not None else self._config.output_dir
        export_transcript(result, self._config.formats, Path(resolved_dir))

        return result
