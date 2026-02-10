"""Configuration loading and management."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from stt.core.pipeline import PipelineConfig

load_dotenv()


@dataclass
class SttConfig:
    model: str = "large-v3"
    language: str = "ru"
    format: str = "json"
    device: str = "cuda"
    compute_type: str = "float16"
    diarization_enabled: bool = True
    min_speakers: int = 1
    max_speakers: int = 8
    output_dir: str = "."
    model_dir: str = "models"
    hf_token: str | None = None
    batch_size: int = 8
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    hallucination_silence_threshold: float = 2.0
    use_subprocess: bool = False
    use_batched: bool = False

    def with_overrides(self, **kwargs: Any) -> SttConfig:
        return replace(self, **kwargs)


def _apply_env_overrides(config: SttConfig) -> SttConfig:
    overrides: dict[str, Any] = {}
    model_dir = os.environ.get("STT_MODEL_DIR")
    if model_dir:
        overrides["model_dir"] = model_dir
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        overrides["hf_token"] = hf_token
    if overrides:
        return replace(config, **overrides)
    return config


def load_config(path: Path | None = None) -> SttConfig:
    if path is None:
        env_path = os.environ.get("STT_CONFIG")
        if env_path:
            path = Path(env_path)

    if path is None:
        cwd_config = Path("config.yaml")
        if cwd_config.exists():
            path = cwd_config

    if path is None or not path.exists():
        return _apply_env_overrides(SttConfig())

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    diarization = data.pop("diarization", None)
    whisper = data.pop("whisper", None)
    kwargs: dict[str, Any] = {}

    for key in (
        "model",
        "language",
        "format",
        "device",
        "compute_type",
        "output_dir",
        "model_dir",
    ):
        if key in data:
            kwargs[key] = data[key]

    if isinstance(diarization, dict):
        if "enabled" in diarization:
            kwargs["diarization_enabled"] = diarization["enabled"]
        if "min_speakers" in diarization:
            kwargs["min_speakers"] = diarization["min_speakers"]
        if "max_speakers" in diarization:
            kwargs["max_speakers"] = diarization["max_speakers"]

    if isinstance(whisper, dict):
        for key in (
            "batch_size",
            "vad_filter",
            "condition_on_previous_text",
            "hallucination_silence_threshold",
            "use_batched",
        ):
            if key in whisper:
                kwargs[key] = whisper[key]

    return _apply_env_overrides(SttConfig(**kwargs))


def resolve_config(
    config: SttConfig,
    *,
    model: str | None = None,
    language: str | None = None,
    format: str | None = None,
    output_dir: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    model_dir: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    no_diarize: bool = False,
) -> SttConfig:
    """Resolve config priority: CLI args > YAML > defaults."""
    overrides: dict[str, Any] = {}
    if model is not None:
        overrides["model"] = model
    if language is not None:
        overrides["language"] = language
    if format is not None:
        overrides["format"] = format
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if device is not None:
        overrides["device"] = device
    if compute_type is not None:
        overrides["compute_type"] = compute_type
    if model_dir is not None:
        overrides["model_dir"] = model_dir
    if min_speakers is not None:
        overrides["min_speakers"] = min_speakers
    if max_speakers is not None:
        overrides["max_speakers"] = max_speakers
    if no_diarize:
        overrides["diarization_enabled"] = False
    if overrides:
        return replace(config, **overrides)
    return config


def build_pipeline_config(
    config: SttConfig,
    *,
    num_speakers: int | None = None,
) -> PipelineConfig:
    """Convert SttConfig to PipelineConfig."""
    from stt.core.pipeline import PipelineConfig

    return PipelineConfig(
        model_size=config.model,
        device=config.device,
        compute_type=config.compute_type,
        language=config.language,
        diarization_enabled=config.diarization_enabled,
        num_speakers=num_speakers,
        min_speakers=config.min_speakers,
        max_speakers=config.max_speakers,
        formats=config.format,
        output_dir=config.output_dir,
        model_dir=config.model_dir,
        hf_token=config.hf_token,
        batch_size=config.batch_size,
        vad_filter=config.vad_filter,
        condition_on_previous_text=config.condition_on_previous_text,
        hallucination_silence_threshold=config.hallucination_silence_threshold,
        use_subprocess=config.use_subprocess,
        use_batched=config.use_batched,
    )
