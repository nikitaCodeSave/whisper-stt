"""Pyannote-based speaker diarization engine."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline

from stt.exceptions import ModelError


@dataclass
class DiarizationTurn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    turns: list[DiarizationTurn]
    num_speakers: int


@dataclass
class DiarizerConfig:
    num_speakers: int | None = None
    min_speakers: int = 1
    max_speakers: int = 8
    model_name: str = "pyannote/speaker-diarization-3.1"
    cache_dir: str | None = None
    hf_token: str | None = None


class PyannoteDiarizer:
    def __init__(self, config: DiarizerConfig) -> None:
        self._config = config
        self._pipeline: Pipeline | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            importlib.import_module("pyannote.audio")
            return True
        except ImportError:
            return False

    def load_model(self) -> None:
        try:
            kwargs: dict[str, Any] = {}
            if self._config.cache_dir is not None:
                kwargs["cache_dir"] = str(
                    Path(self._config.cache_dir).expanduser().resolve()
                )
            if self._config.hf_token:
                kwargs["token"] = self._config.hf_token
            self._pipeline = Pipeline.from_pretrained(
                self._config.model_name, **kwargs,
            )
        except Exception as e:
            raise ModelError(f"Failed to load diarization model: {e}") from e

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def diarize(self, audio_path: str) -> DiarizationResult:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        kwargs: dict[str, Any] = {}
        if self._config.num_speakers is not None:
            kwargs["num_speakers"] = self._config.num_speakers
        else:
            kwargs["min_speakers"] = self._config.min_speakers
            kwargs["max_speakers"] = self._config.max_speakers
        result = self._pipeline(audio_path, **kwargs)
        # pyannote 4.x returns DiarizeOutput; extract Annotation from it
        annotation = getattr(result, "speaker_diarization", result)
        turns: list[DiarizationTurn] = []
        speakers: set[str] = set()
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(start=turn.start, end=turn.end, speaker=speaker)
            )
            speakers.add(speaker)
        return DiarizationResult(turns=turns, num_speakers=len(speakers))
