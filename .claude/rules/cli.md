---
paths:
  - "stt/cli/**"
---

# Правила CLI

## Typer синтаксис

```python
from typing import Annotated
import typer

def command(
    audio_file: Annotated[Path, typer.Argument(help="...")],
    model: Annotated[str, typer.Option("--model", "-m", help="...")] = "large-v3",
) -> None:
```

## Exit codes

Маппинг исключений → exit codes:
- `AudioValidationError` → `ExitCode.ERROR_FILE` (3)
- `AudioPreprocessError` → `ExitCode.ERROR_FILE` (3)
- `GpuError` → `ExitCode.ERROR_GPU` (5)
- `ModelError` → `ExitCode.ERROR_MODEL` (4)
- Batch partial → `ExitCode.PARTIAL_SUCCESS` (10)

Выход через `raise typer.Exit(code=ExitCode.XXX)`, NEVER `sys.exit()`.

## Ошибки

```python
typer.echo(f"Error: {e}", err=True)
raise typer.Exit(code=ExitCode.ERROR_FILE)
```

## CLI контракт

Следовать: [docs/api.md](../../docs/api.md)
