"""Transcribe command for the STT CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from stt.config import build_pipeline_config, load_config, resolve_config
from stt.core.audio import validate_audio_file
from stt.core.pipeline import TranscriptionPipeline
from stt.exceptions import AudioPreprocessError, AudioValidationError, GpuError, ModelError
from stt.exit_codes import ExitCode


def transcribe_cmd(
    audio_file: Annotated[
        Path, typer.Argument(help="Path to the audio file."),
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
    model_dir: Annotated[
        str | None,
        typer.Option(
            "--model-dir",
            help="Directory for model storage.",
        ),
    ] = None,
) -> None:
    """Transcribe a single audio file."""
    # Validate audio file
    try:
        validate_audio_file(audio_file)
    except AudioValidationError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=ExitCode.ERROR_FILE) from None

    # Validate speaker hint conflicts
    if num_speakers is not None and (
        min_speakers is not None or max_speakers is not None
    ):
        typer.echo(
            "Error: --num-speakers cannot be used with "
            "--min-speakers or --max-speakers.",
            err=True,
        )
        raise typer.Exit(code=ExitCode.ERROR_ARGS)

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

    try:
        pipeline = TranscriptionPipeline(config)
        pipeline.run(str(audio_file))
    except AudioPreprocessError as e:
        typer.echo(f"Audio preprocessing error: {e}", err=True)
        raise typer.Exit(code=ExitCode.ERROR_FILE) from None
    except GpuError as e:
        typer.echo(f"GPU error: {e}", err=True)
        raise typer.Exit(code=ExitCode.ERROR_GPU) from None
    except ModelError as e:
        typer.echo(f"Model error: {e}", err=True)
        raise typer.Exit(code=ExitCode.ERROR_MODEL) from None
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=ExitCode.ERROR_GENERAL) from None
