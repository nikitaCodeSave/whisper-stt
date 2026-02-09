# Local STT Service

Локальный CLI для транскрипции аудио (русский + en) с диаризацией спикеров. Python 3.11+, faster-whisper, pyannote.audio 4.x, Typer. GPU-only (>=10GB VRAM).

## Команды

```bash
# Тесты (220 unit-тестов, все мокают ML-модели, GPU не нужен)
python -m pytest tests/unit/ -v --tb=short
python -m pytest tests/unit/test_aligner.py -v        # один модуль
python -m pytest tests/unit/ --cov=stt --cov-report=term-missing

# Линтинг
ruff check stt/ tests/
mypy stt/

# Запуск CLI
stt transcribe meeting.mp3 --format json,txt,srt --output ./results/
stt transcribe call.wav --no-diarize --format txt
stt batch ./recordings/ -r --pattern "*.mp3" --format json --output ./out/
stt models list
stt models download large-v3 --model-dir models

# Установка dev-окружения
pip install -e ".[dev]"
```

**IMPORTANT:** После КАЖДОГО изменения кода запускать `python -m pytest tests/unit/ -v --tb=short && ruff check stt/ tests/`.

## Архитектура

Точка входа: `stt.cli.app:app` (Typer). Пакет `stt/`.

```
CLI (cli/)  →  Pipeline (core/pipeline.py)  →  Exporters (exporters/)
                    |
         Transcriber + Diarizer + Aligner  (core/)
                    |
         Data Models + Config + Exceptions  (корень stt/)
```

**CRITICAL: Sequential VRAM.** Transcriber (~8GB) и Diarizer (~6GB) НИКОГДА не загружены одновременно:
```
transcriber.load_model() → transcribe() → unload_model()
diarizer.load_model()    → diarize()    → unload_model()
```
Это позволяет работать на GPU с 10GB. NEVER загружать обе модели одновременно.

Подробнее: [dev-docs/architecture.md](dev-docs/architecture.md)

## Конвенции кода

- ALWAYS: `from __future__ import annotations` в первой строке каждого .py файла
- ALWAYS: type hints на всех функциях и методах (mypy strict: `disallow_untyped_defs`)
- ALWAYS: `X | None` вместо `Optional[X]`
- ALWAYS: dataclasses (не Pydantic), иммутабельность через `dataclasses.replace()`
- ALWAYS: docstring — одна строка: `"""Описание модуля."""`
- Line length: 100 символов (ruff)
- Lint rules: E, F, I (isort), UP (pyupgrade), B (bugbear)
- Typer CLI: `Annotated[type, typer.Option(...)]` синтаксис
- Ошибки: кастомные исключения из `stt/exceptions.py` → exit codes из `stt/exit_codes.py`
- Circular imports: `TYPE_CHECKING` guard (пример: `stt/config.py`)
- Тесты: mock ВСЕ ML-модели, `@patch("stt.core.transcriber.WhisperModel")` и т.д.

## Gotchas

- **pyannote 4.x API:** `Pipeline.from_pretrained(model, token=...)` (НЕ `use_auth_token`). Результат вызова — `DiarizeOutput`, аннотация через `.speaker_diarization`
- **Audio preprocessing:** ВСЕ форматы конвертируются в WAV 16kHz mono через ffmpeg перед обработкой (`stt/core/audio.py`)
- **Config priority:** CLI флаги > `STT_CONFIG` env > `./config.yaml` > defaults. `HF_TOKEN` загружается из `.env` автоматически (python-dotenv)
- **Exit codes:** Специфичные: 2=args, 3=file, 4=model, 5=GPU, 10=partial batch. Использовать `raise typer.Exit(code=ExitCode.XXX)`, NEVER `sys.exit()`
- **Экспорт в stdout:** JSON без `--output` идёт в stdout (для pipe). С `--output` — создаёт файлы
- **Batch partial success:** Если часть файлов упала — `ExitCode.PARTIAL_SUCCESS` (10), не ERROR
- **NEVER хардкодить HF_TOKEN** — только через `.env` / env vars через `config.py`

## Документация

| Когда | Что читать |
|-------|-----------|
| CLI контракт: флаги, коды возврата | [docs/api.md](docs/api.md) |
| JSON/TXT/SRT форматы и schema | [docs/data-formats.md](docs/data-formats.md) |
| Бизнес-цели, scope, метрики | [docs/PRD.md](docs/PRD.md) |
| Функциональные требования, AC | [docs/SPEC.md](docs/SPEC.md) |
| Почему выбрали faster-whisper/pyannote | [docs/decisions/ADR-001-tech-stack.md](docs/decisions/ADR-001-tech-stack.md) |
| Архитектура: слои, VRAM, alignment | [dev-docs/architecture.md](dev-docs/architecture.md) |
| Setup, env vars, TDD, расширение | [dev-docs/dev-guide.md](dev-docs/dev-guide.md) |
| Тесты: стратегия моков, coverage | [dev-docs/testing.md](dev-docs/testing.md) |
| API модулей (Segment, Transcriber...) | [dev-docs/module-reference.md](dev-docs/module-reference.md) |

## Границы для AI

### DO

- Следовать CLI-контракту из [docs/api.md](docs/api.md) и JSON-schema из [docs/data-formats.md](docs/data-formats.md)
- Разделять STT-движок и диаризатор как независимые модули (`core/transcriber.py` и `core/diarizer.py`)
- Использовать type hints везде (mypy strict)
- Обрабатывать ошибки через `stt/exceptions.py` → exit codes
- Фиксировать версии зависимостей в `pyproject.toml`
- Писать тесты ПЕРЕД реализацией (TDD)

### DO NOT

- NEVER отправлять данные на внешние серверы (полностью локальный сервис)
- NEVER хардкодить HF_TOKEN (только через `.env` / env vars)
- NEVER смешивать логику транскрипции и диаризации в одном модуле
- NEVER использовать оригинальный OpenAI Whisper (только faster-whisper)
- NEVER использовать NeMo Sortformer (CC-BY-NC лицензия, non-commercial)
- NEVER загружать Transcriber и Diarizer в GPU одновременно

## Структура

```
stt/
  cli/           # Typer CLI: app.py (entry), transcribe.py, batch.py, models_cmd.py
  core/          # ML pipeline: transcriber, diarizer, aligner, pipeline, audio, batch
  exporters/     # JSON/TXT/SRT export + dispatch
  config.py      # SttConfig, load_config(), load_dotenv()
  data_models.py # Segment, TranscriptMetadata, TranscriptResult
  exceptions.py  # AudioValidationError, GpuError, ModelError, AudioPreprocessError
  exit_codes.py  # ExitCode IntEnum (0,1,2,3,4,5,10)
tests/
  unit/          # 220 тестов (21 файл), все мокают ML-модели
  integration/   # E2E (TODO)
```

Полная структура и расширение (добавление экспортера/диаризатора): [dev-docs/dev-guide.md](dev-docs/dev-guide.md)
