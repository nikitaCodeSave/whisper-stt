# Руководство разработчика

## Требования

- Python 3.11+ (рекомендуется 3.12)
- NVIDIA GPU с >=10 GB VRAM (для large-v3)
- ffmpeg (системный)
- HuggingFace токен (для скачивания моделей pyannote)

## Установка

```bash
# Создать виртуальное окружение
python3.12 -m venv .venv
source .venv/bin/activate

# Установить с dev-зависимостями
pip install -e ".[dev]"

# Настроить окружение
cp config.yaml.example config.yaml
echo "HF_TOKEN=hf_your_token_here" > .env   # токен с huggingface.co/settings/tokens
mkdir -p models output

# Скачать модели
stt models download large-v3 --model-dir models

# Проверить установку
stt --version        # stt 1.0.0
python -m pytest     # 220 passed
ruff check stt/      # All checks passed
mypy stt/            # Success: no issues found
```

> **Примечание:** `python-dotenv` автоматически загружает переменные из `.env` — ручной `source .env` не нужен.

## Структура проекта

```
stt/
  __init__.py              # __version__
  exit_codes.py            # ExitCode IntEnum
  exceptions.py            # AudioValidationError, GpuError, ModelError
  data_models.py           # Segment, TranscriptMetadata, TranscriptResult
  config.py                # SttConfig, load_config(), load_dotenv()
  cli/
    app.py                 # Typer app, entry point
    transcribe.py          # stt transcribe
    batch.py               # stt batch
    models_cmd.py          # stt models {list,info,download}
  core/
    audio.py               # validate_audio_file()
    speaker_hints.py       # SpeakerHints, validate_speaker_hints()
    transcriber.py         # Transcriber (faster-whisper)
    diarizer.py            # PyannoteDiarizer (pyannote 4.x)
    aligner.py             # align_segments()
    pipeline.py            # TranscriptionPipeline
    batch.py               # discover_audio_files(), BatchRunner
  exporters/
    __init__.py            # export_transcript() dispatcher
    json_export.py         # export_json()
    txt_export.py          # export_txt()
    srt_export.py          # export_srt()
tests/
  unit/                    # 21 тестовый файл, 220 тестов
  integration/             # E2E с tiny моделью (TODO)
  fixtures/                # Тестовые аудиофайлы (TODO)
```

## Добавление нового экспортера

1. Создать `stt/exporters/new_export.py`:
```python
from typing import IO
from stt.data_models import TranscriptResult

def export_new(result: TranscriptResult, output: IO[str]) -> None:
    for seg in result.segments:
        output.write(f"{seg.start} {seg.text}\n")
```

2. Зарегистрировать в `stt/exporters/__init__.py`:
```python
from stt.exporters.new_export import export_new

_EXPORTERS["new"] = export_new
```

3. Написать тесты в `tests/unit/test_new_export.py`

## Добавление нового диаризатора (v2)

`PyannoteDiarizer` уже использует паттерн с `load_model()`/`unload_model()`/`diarize()`. Для добавления альтернативы (например NeMo):

1. Реализовать класс с тем же интерфейсом в `stt/core/nemo_diarizer.py`
2. Вынести Protocol:
```python
class DiarizationProtocol(Protocol):
    def load_model(self) -> None: ...
    def unload_model(self) -> None: ...
    def diarize(self, audio_path: str) -> DiarizationResult: ...
    @staticmethod
    def is_available() -> bool: ...
```
3. Использовать в `pipeline.py` через фабрику

## Конвенции кода

- **Type hints** везде (Typer требует)
- **Dataclasses** для конфигураций и моделей данных
- **Иммутабельность**: `aligner` создаёт новые Segment через `replace()`
- **Line length**: 100 символов (ruff)
- **Imports**: `from __future__ import annotations` в каждом файле
- **Lint rules**: E, F, I (isort), UP (pyupgrade), B (bugbear)

## Рабочий процесс (TDD)

1. Написать тест (RED)
2. Реализовать минимальный код (GREEN)
3. Отрефакторить (REFACTOR)
4. `pytest` + `ruff check` после каждого изменения

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `HF_TOKEN` | HuggingFace токен (для pyannote) | — |
| `STT_MODEL_DIR` | Директория хранения моделей | `models` |
| `STT_CONFIG` | Путь к файлу конфигурации | `./config.yaml` |
| `CUDA_VISIBLE_DEVICES` | GPU для использования | все доступные |

Все переменные загружаются автоматически из `.env` через `python-dotenv`.

## Полезные команды

```bash
# Транскрипция одного файла
stt transcribe meeting.mp3 --format json,txt,srt --output ./results/

# Batch с рекурсией
stt batch ./recordings/ -r --pattern "*.mp3" --format json --output ./out/

# Без диаризации (быстрее)
stt transcribe call.wav --no-diarize --format txt

# Конкретное число спикеров
stt transcribe meeting.m4a --num-speakers 3

# CPU-режим (без GPU)
stt transcribe test.wav --device cpu --compute-type int8

# Управление моделями
stt models list
stt models info large-v3
stt models download large-v3 --model-dir models
```
