"""Batch processing command for the STT CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from stt.config import build_pipeline_config, load_config, resolve_config
from stt.core.batch import BatchRunner, discover_audio_files
from stt.exit_codes import ExitCode


def batch_cmd(
    input_dir: Annotated[
        Path,
        typer.Argument(help="Directory with audio files."),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Whisper model size."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Audio language."),
    ] = None,
    format: Annotated[
        str | None,
        typer.Option(
            "--format", "-f",
            help="Output format(s): json,txt,srt.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory."),
    ] = None,
    no_diarize: Annotated[
        bool,
        typer.Option(
            "--no-diarize",
            help="Disable speaker diarization.",
        ),
    ] = False,
    num_speakers: Annotated[
        int | None,
        typer.Option(
            "--num-speakers",
            help="Exact number of speakers.",
        ),
    ] = None,
    min_speakers: Annotated[
        int | None,
        typer.Option(
            "--min-speakers",
            help="Minimum number of speakers.",
        ),
    ] = None,
    max_speakers: Annotated[
        int | None,
        typer.Option(
            "--max-speakers",
            help="Maximum number of speakers.",
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device: cuda or cpu."),
    ] = None,
    compute_type: Annotated[
        str | None,
        typer.Option("--compute-type", help="Compute type."),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive", "-r",
            help="Process subdirectories.",
        ),
    ] = False,
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern", help="Glob pattern for audio files.",
        ),
    ] = "*",
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help="Skip already processed files.",
        ),
    ] = False,
    model_dir: Annotated[
        str | None,
        typer.Option(
            "--model-dir",
            help="Directory for model storage.",
        ),
    ] = None,
) -> None:
    """Batch process audio files in a directory."""
    if not input_dir.exists():
        typer.echo(
            f"Error: Directory not found: {input_dir}", err=True,
        )
        raise typer.Exit(code=ExitCode.ERROR_FILE)

    files = discover_audio_files(
        input_dir, recursive=recursive, pattern=pattern,
    )

    if not files:
        typer.echo("No audio files found.", err=True)
        raise typer.Exit(code=0)

    # Load YAML config, resolve CLI overrides, build pipeline config
    stt_config = resolve_config(
        load_config(),
        model=model,
        language=language,
        format=format,
        output_dir=str(output) if output is not None else None,
        device=device,
        compute_type=compute_type,
        model_dir=model_dir,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        no_diarize=no_diarize,
    )
    config = build_pipeline_config(stt_config, num_speakers=num_speakers)

    resolved_output = Path(stt_config.output_dir)
    runner = BatchRunner(config, skip_existing=skip_existing)
    result = runner.run(
        files,
        resolved_output,
        input_base=input_dir if recursive else None,
    )

    typer.echo(
        f"Processed {result.succeeded}/{result.total} files.",
        err=True,
    )
    if result.errors:
        for path, err in result.errors:
            typer.echo(f"  Failed: {path} - {err}", err=True)

    raise typer.Exit(code=result.exit_code)
