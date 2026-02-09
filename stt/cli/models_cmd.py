"""Models subcommands for the STT CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from stt.config import load_config
from stt.exit_codes import ExitCode

models_app = typer.Typer(name="models", help="Manage STT models.")

AVAILABLE_MODELS: dict[str, dict[str, str]] = {
    "tiny": {
        "size": "75 MB",
        "vram": "~1 GB",
        "description": "Fastest, lowest quality.",
    },
    "base": {
        "size": "142 MB",
        "vram": "~1 GB",
        "description": "Fast, basic quality.",
    },
    "small": {
        "size": "466 MB",
        "vram": "~2 GB",
        "description": "Good balance of speed and quality.",
    },
    "medium": {
        "size": "1.5 GB",
        "vram": "~5 GB",
        "description": "High quality, slower.",
    },
    "large-v3": {
        "size": "3.1 GB",
        "vram": "~10 GB",
        "description": "Best quality, requires good GPU.",
    },
    "large-v3-turbo": {
        "size": "1.6 GB",
        "vram": "~6 GB",
        "description": "Fast large model variant.",
    },
}


@models_app.command("list")
def list_models() -> None:
    """List available Whisper models."""
    typer.echo(
        f"{'Model':<18} {'Size':<10} {'VRAM':<10} Description"
    )
    typer.echo("-" * 70)
    for name, info in AVAILABLE_MODELS.items():
        typer.echo(
            f"{name:<18} {info['size']:<10} {info['vram']:<10} "
            f"{info['description']}"
        )


@models_app.command("info")
def model_info(
    model: str = typer.Argument(..., help="Model name."),
) -> None:
    """Show details for a specific model."""
    if model not in AVAILABLE_MODELS:
        typer.echo(f"Error: Unknown model '{model}'.", err=True)
        raise typer.Exit(code=ExitCode.ERROR_MODEL)

    info = AVAILABLE_MODELS[model]
    typer.echo(f"Model:       {model}")
    typer.echo(f"Size:        {info['size']}")
    typer.echo(f"VRAM:        {info['vram']}")
    typer.echo(f"Description: {info['description']}")


@models_app.command("download")
def download_model(
    model: str = typer.Argument(
        ..., help="Model name to download.",
    ),
    model_dir: Annotated[
        str | None,
        typer.Option(
            "--model-dir",
            help="Directory for model storage.",
        ),
    ] = None,
) -> None:
    """Download a Whisper model for offline use."""
    if model not in AVAILABLE_MODELS:
        typer.echo(f"Error: Unknown model '{model}'.", err=True)
        raise typer.Exit(code=ExitCode.ERROR_MODEL)

    stt_config = load_config()
    resolved_model_dir = model_dir if model_dir is not None else stt_config.model_dir

    # Download Whisper model
    typer.echo(f"Downloading Whisper model '{model}'...")
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        download_root = str(Path(resolved_model_dir).expanduser().resolve())
        WhisperModel(
            model,
            device="cpu",
            compute_type="int8",
            download_root=download_root,
        )
        typer.echo(f"Whisper model '{model}' downloaded to {download_root}.")
    except Exception as e:
        typer.echo(f"Error downloading Whisper model: {e}", err=True)
        raise typer.Exit(code=ExitCode.ERROR_MODEL) from None

    # Download pyannote model if HF_TOKEN is set
    hf_token = stt_config.hf_token
    if hf_token:
        typer.echo("Downloading pyannote diarization model...")
        try:
            from pyannote.audio import Pipeline

            cache_dir = str(Path(resolved_model_dir).expanduser().resolve())
            Pipeline.from_pretrained(  # type: ignore[call-arg]
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
                cache_dir=cache_dir,
            )
            typer.echo(f"Pyannote model downloaded to {cache_dir}.")
        except Exception as e:
            typer.echo(
                f"Error downloading pyannote model: {e}", err=True,
            )
            raise typer.Exit(code=ExitCode.ERROR_MODEL) from None
    else:
        typer.echo(
            "HF_TOKEN not set — skipping pyannote model download. "
            "Set HF_TOKEN to download the diarization model.",
        )
