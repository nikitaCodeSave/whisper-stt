"""Main CLI application."""

from __future__ import annotations

import typer

from stt import __version__
from stt.cli.batch import batch_cmd
from stt.cli.models_cmd import models_app
from stt.cli.transcribe import transcribe_cmd
from stt.core.gpu_utils import configure_cuda_allocator

app = typer.Typer(
    name="stt",
    help="Local STT service with speaker diarization.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"stt {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Local STT service with speaker diarization."""
    configure_cuda_allocator()


app.command("transcribe")(transcribe_cmd)
app.command("batch")(batch_cmd)
app.add_typer(models_app)
