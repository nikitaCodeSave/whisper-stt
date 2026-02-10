"""Whisper-based transcription engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel

from stt.core.gpu_utils import cleanup_gpu_memory
from stt.data_models import Segment
from stt.exceptions import CudaOomError, GpuError, ModelError, TranscriptionError


def _map_confidence(avg_logprob: float) -> float:
    """Convert avg_logprob to confidence 0-1 using exp(), clamped."""
    return min(max(math.exp(avg_logprob), 0.0), 1.0)


@dataclass
class TranscriberConfig:
    model_size: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    model_dir: str | None = None
    language: str = "ru"
    batch_size: int = 8
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    hallucination_silence_threshold: float = 2.0
    use_batched: bool = False


class Transcriber:
    def __init__(self, config: TranscriberConfig) -> None:
        self._config = config
        self._model: WhisperModel | None = None
        self._batched: BatchedInferencePipeline | None = None

    def load_model(self) -> None:
        if self._config.device == "cuda" and not torch.cuda.is_available():
            raise GpuError("CUDA is not available")
        try:
            kwargs: dict[str, Any] = {
                "device": self._config.device,
                "compute_type": self._config.compute_type,
            }
            if self._config.model_dir is not None:
                kwargs["download_root"] = str(
                    Path(self._config.model_dir).expanduser().resolve()
                )
            self._model = WhisperModel(
                self._config.model_size,
                **kwargs,
            )
            self._batched = BatchedInferencePipeline(model=self._model)
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}") from e

    def unload_model(self) -> None:
        self._batched = None
        self._model = None
        cleanup_gpu_memory("transcriber_unload")

    def transcribe(self, audio_path: str) -> list[Segment]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        try:
            if self._config.use_batched:
                if self._batched is None:
                    raise RuntimeError("Model not loaded. Call load_model() first.")
                segments_iter, _info = self._batched.transcribe(
                    audio_path,
                    language=self._config.language,
                    batch_size=self._config.batch_size,
                    # without_timestamps=False is required so that the model
                    # generates timestamp tokens; otherwise the entire VAD
                    # chunk is returned as a single segment.
                    without_timestamps=False,
                    vad_filter=self._config.vad_filter,
                    vad_parameters={"min_silence_duration_ms": 500},
                    condition_on_previous_text=(
                        self._config.condition_on_previous_text
                    ),
                    hallucination_silence_threshold=(
                        self._config.hallucination_silence_threshold
                    ),
                )
            else:
                segments_iter, _info = self._model.transcribe(
                    audio_path,
                    language=self._config.language,
                    vad_filter=self._config.vad_filter,
                    vad_parameters={"min_silence_duration_ms": 500},
                    condition_on_previous_text=(
                        self._config.condition_on_previous_text
                    ),
                    hallucination_silence_threshold=(
                        self._config.hallucination_silence_threshold
                    ),
                )
            result = []
            for seg in segments_iter:
                result.append(
                    Segment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                        confidence=_map_confidence(seg.avg_logprob),
                    )
                )
        except torch.cuda.OutOfMemoryError as e:
            raise CudaOomError(f"CUDA OOM during transcription: {e}") from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise CudaOomError(f"CUDA OOM during transcription: {e}") from e
            raise TranscriptionError(f"Transcription failed: {e}") from e
        return result
