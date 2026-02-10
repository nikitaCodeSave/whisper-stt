"""Subprocess isolation for pipeline stages."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import asdict
from typing import Any

from stt.data_models import Segment


def _transcribe_worker(
    config_dict: dict[str, Any], audio_path: str, queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Run transcription in a child process."""
    try:
        from stt.core.transcriber import Transcriber, TranscriberConfig

        config = TranscriberConfig(**config_dict)
        transcriber = Transcriber(config)
        transcriber.load_model()
        try:
            segments = transcriber.transcribe(audio_path)
        finally:
            transcriber.unload_model()
        queue.put({"status": "ok", "segments": [asdict(s) for s in segments]})
    except Exception as e:
        queue.put({"status": "error", "error": f"{type(e).__name__}: {e}"})


def _diarize_worker(
    config_dict: dict[str, Any], audio_path: str, queue: mp.Queue,  # type: ignore[type-arg]
) -> None:
    """Run diarization in a child process."""
    try:
        from stt.core.diarizer import (
            DiarizerConfig,
            PyannoteDiarizer,
        )

        config = DiarizerConfig(**config_dict)
        diarizer = PyannoteDiarizer(config)
        diarizer.load_model()
        try:
            result = diarizer.diarize(audio_path)
        finally:
            diarizer.unload_model()
        queue.put({
            "status": "ok",
            "turns": [asdict(t) for t in result.turns],
            "num_speakers": result.num_speakers,
        })
    except Exception as e:
        queue.put({"status": "error", "error": f"{type(e).__name__}: {e}"})


def run_transcription_subprocess(
    config_dict: dict[str, Any],
    audio_path: str,
    timeout: float | None = None,
) -> list[Segment]:
    """Run transcription in a subprocess for full GPU memory isolation."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue[dict[str, Any]] = ctx.Queue()  # type: ignore[type-arg]
    process = ctx.Process(
        target=_transcribe_worker, args=(config_dict, audio_path, queue),
    )
    process.start()
    try:
        result = queue.get(timeout=timeout)
    except Exception as e:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        raise RuntimeError(f"Subprocess transcription failed: {e}") from e
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
    if result["status"] == "error":
        raise RuntimeError(result["error"])
    return [Segment(**s) for s in result["segments"]]


def run_diarization_subprocess(
    config_dict: dict[str, Any],
    audio_path: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run diarization in a subprocess for full GPU memory isolation."""
    ctx = mp.get_context("spawn")
    queue: mp.Queue[dict[str, Any]] = ctx.Queue()  # type: ignore[type-arg]
    process = ctx.Process(
        target=_diarize_worker, args=(config_dict, audio_path, queue),
    )
    process.start()
    try:
        result = queue.get(timeout=timeout)
    except Exception as e:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        raise RuntimeError(f"Subprocess diarization failed: {e}") from e
    process.join(timeout=30)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
    if result["status"] == "error":
        raise RuntimeError(result["error"])
    return result
