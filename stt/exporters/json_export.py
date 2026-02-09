"""JSON exporter for transcription results."""

from __future__ import annotations

import json
from typing import IO

from stt.data_models import TranscriptResult


def export_json(result: TranscriptResult, output: IO[str]) -> None:
    """Write transcription result as JSON to the given output stream."""
    meta = result.metadata
    metadata_dict = {
        "format_version": meta.format_version,
        "source_file": meta.source_file,
        "duration_seconds": meta.duration_seconds,
        "language": meta.language,
        "model": meta.model,
        "diarization": meta.diarization,
        "num_speakers": meta.num_speakers,
        "processing_time_seconds": meta.processing_time_seconds,
        "created_at": meta.created_at.isoformat(),
    }

    segments_list = []
    for seg in result.segments:
        seg_dict: dict[str, object] = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        if seg.speaker is not None:
            seg_dict["speaker"] = seg.speaker
        if seg.confidence is not None:
            seg_dict["confidence"] = seg.confidence
        segments_list.append(seg_dict)

    data = {
        "metadata": metadata_dict,
        "segments": segments_list,
        "full_text": result.full_text,
    }

    json.dump(data, output, indent=2, ensure_ascii=False)
