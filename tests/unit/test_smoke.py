"""Sprint 0: Smoke tests."""

from typer.testing import CliRunner

from stt import __version__
from stt.cli.app import app


def test_import_stt() -> None:
    """stt package is importable and has a version."""
    assert __version__ == "1.0.0"


def test_stt_help_not_crashes() -> None:
    """stt --help exits cleanly."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "stt" in result.output.lower()
