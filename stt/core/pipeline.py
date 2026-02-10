"""Transcription pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from stt.core.aligner import align_segments
from stt.core.audio import preprocess_audio, validate_audio_file
from stt.core.diarizer import (
    DiarizationResult,
    DiarizationTurn,
    DiarizerConfig,
    PyannoteDiarizer,
)
from stt.core.gpu_utils import cleanup_gpu_memory, log_gpu_memory
from stt.core.subprocess_runner import (
    run_diarization_subprocess,
    run_transcription_subprocess,
)
from stt.core.transcriber import Transcriber, TranscriberConfig
from stt.data_models import TranscriptMetadata, TranscriptResult
from stt.exporters import export_transcript

logger = logging.getLogger(__name__)


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
    batch_size: int = 8
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    hallucination_silence_threshold: float = 2.0
    use_subprocess: bool = False
    use_batched: bool = False


class TranscriptionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self, audio_path: str, output_dir: str | None = None) -> TranscriptResult:
        start_time = time.monotonic()

        # 1. Validate audio
        validate_audio_file(Path(audio_path))

        # 2. Preprocess: convert to WAV 16kHz mono for whisper & pyannote
        preprocessed = preprocess_audio(Path(audio_path))
        preprocessed_path = str(preprocessed.path)

        try:
            # 3. Transcribe: load, run, unload (free VRAM)
            transcriber_config = TranscriberConfig(
                model_size=self._config.model_size,
                device=self._config.device,
                compute_type=self._config.compute_type,
                model_dir=self._config.model_dir,
                language=self._config.language,
                batch_size=self._config.batch_size,
                vad_filter=self._config.vad_filter,
                condition_on_previous_text=self._config.condition_on_previous_text,
                hallucination_silence_threshold=(
                    self._config.hallucination_silence_threshold
                ),
                use_batched=self._config.use_batched,
            )

            if self._config.use_subprocess:
                t1 = time.monotonic()
                segments = run_transcription_subprocess(
                    asdict(transcriber_config), preprocessed_path,
                )
                t2 = time.monotonic()
                logger.info(
                    "Transcription (subprocess) completed in %.1fs (%d segments)",
                    t2 - t1, len(segments),
                )
            else:
                transcriber = Transcriber(transcriber_config)
                try:
                    log_gpu_memory("before_transcriber_load")
                    transcriber.load_model()
                    log_gpu_memory("after_transcriber_load")
                    t1 = time.monotonic()
                    segments = transcriber.transcribe(preprocessed_path)
                    t2 = time.monotonic()
                    logger.info(
                        "Transcription completed in %.1fs (%d segments)",
                        t2 - t1, len(segments),
                    )
                finally:
                    try:
                        transcriber.unload_model()
                    except Exception:
                        logger.exception("Failed to unload transcriber")
                cleanup_gpu_memory("after_transcriber_unload")

            # 4. Diarize if enabled: load, run, unload (free VRAM)
            num_speakers = 0
            if self._config.diarization_enabled:
                diarizer_config = DiarizerConfig(
                    num_speakers=self._config.num_speakers,
                    min_speakers=self._config.min_speakers,
                    max_speakers=self._config.max_speakers,
                    cache_dir=self._config.model_dir,
                    hf_token=self._config.hf_token,
                )

                if self._config.use_subprocess:
                    t3 = time.monotonic()
                    raw = run_diarization_subprocess(
                        asdict(diarizer_config), preprocessed_path,
                    )
                    t4 = time.monotonic()
                    logger.info("Diarization (subprocess) completed in %.1fs", t4 - t3)
                    diarization_result = DiarizationResult(
                        turns=[DiarizationTurn(**t) for t in raw["turns"]],
                        num_speakers=raw["num_speakers"],
                    )
                else:
                    diarizer = PyannoteDiarizer(diarizer_config)
                    try:
                        log_gpu_memory("before_diarizer_load")
                        diarizer.load_model()
                        log_gpu_memory("after_diarizer_load")
                        t3 = time.monotonic()
                        diarization_result = diarizer.diarize(preprocessed_path)
                        t4 = time.monotonic()
                        logger.info("Diarization completed in %.1fs", t4 - t3)
                    finally:
                        try:
                            diarizer.unload_model()
                        except Exception:
                            logger.exception("Failed to unload diarizer")
                    cleanup_gpu_memory("after_diarizer_unload")

                # 5. Align segments with diarization
                segments = align_segments(segments, diarization_result)
                num_speakers = diarization_result.num_speakers
        finally:
            preprocessed.cleanup()

        # 6. Build result
        elapsed = time.monotonic() - start_time
        logger.info("Total pipeline: %.1fs", elapsed)
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

        # 7. Export
        resolved_dir = output_dir if output_dir is not None else self._config.output_dir
        export_transcript(result, self._config.formats, Path(resolved_dir))

        return result
