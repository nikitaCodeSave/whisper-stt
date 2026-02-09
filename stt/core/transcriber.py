"""Whisper-based transcription engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel

from stt.data_models import Segment
from stt.exceptions import GpuError, ModelError


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


class Transcriber:
    def __init__(self, config: TranscriberConfig) -> None:
        self._config = config
        self._model: WhisperModel | None = None

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
        except Exception as e:
            raise ModelError(f"Failed to load model: {e}") from e

    def unload_model(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self, audio_path: str) -> list[Segment]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        segments_iter, _info = self._model.transcribe(audio_path, language=self._config.language)
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
        return sorted(result, key=lambda s: s.start)
