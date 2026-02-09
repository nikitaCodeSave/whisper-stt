"""Export dispatch for transcription results."""

from __future__ import annotations

from collections.abc import Callable
from io import StringIO
from pathlib import Path

from stt.data_models import TranscriptResult
from stt.exporters.json_export import export_json
from stt.exporters.srt_export import export_srt
from stt.exporters.txt_export import export_txt

_EXPORTERS: dict[str, Callable[..., None]] = {
    "json": export_json,
    "txt": export_txt,
    "srt": export_srt,
}


def export_transcript(
    result: TranscriptResult,
    formats: str,
    output_dir: Path | None = None,
) -> str | None:
    """Export transcription result in the specified formats."""
    format_list = [f.strip() for f in formats.split(",")]

    for fmt in format_list:
        if fmt not in _EXPORTERS:
            raise ValueError(f"Unknown export format: {fmt!r}")

    if output_dir is None and format_list == ["json"]:
        buf = StringIO()
        export_json(result, buf)
        return buf.getvalue()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(result.metadata.source_file).stem
        for fmt in format_list:
            out_path = output_dir / f"{stem}.{fmt}"
            with open(out_path, "w", encoding="utf-8") as f:
                _EXPORTERS[fmt](result, f)

    return None
