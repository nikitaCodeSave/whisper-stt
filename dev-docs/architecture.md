# Архитектура: Local STT Service

## Обзор

Сервис реализован как CLI-приложение на Python с модульной архитектурой. Основной поток данных — последовательный pipeline: аудиофайл проходит через STT (faster-whisper), диаризацию (pyannote), alignment и экспортируется в JSON/TXT/SRT.

## Принцип управления VRAM

Ключевое архитектурное решение — **sequential VRAM management**. STT-модель и диаризатор не находятся в памяти GPU одновременно:

```
Transcriber.load_model()     ← загрузка в VRAM (~8 GB)
segments = transcriber.transcribe(audio)
Transcriber.unload_model()   ← VRAM освобождён

Diarizer.load_model()        ← загрузка в VRAM (~6 GB)
diarization = diarizer.diarize(audio)
Diarizer.unload_model()      ← VRAM освобождён
```

Это позволяет работать на GPU с 10 GB VRAM вместо 24 GB.

## Слои системы

```
┌─────────────────────────────────────────────────┐
│                  CLI (Typer)                     │
│  app.py → transcribe.py, batch.py, models_cmd   │
├─────────────────────────────────────────────────┤
│              Pipeline Orchestrator               │
│  pipeline.py — координация STT → Diarize → Align│
├────────────┬────────────┬───────────────────────┤
│ Transcriber│  Diarizer  │       Aligner         │
│ (faster-   │ (pyannote) │  align_segments()     │
│  whisper)  │            │  _compute_overlap()   │
├────────────┴────────────┴───────────────────────┤
│               Data Models                        │
│  Segment, TranscriptMetadata, TranscriptResult   │
├─────────────────────────────────────────────────┤
│               Exporters                          │
│  JSON, TXT, SRT + dispatch                       │
├─────────────────────────────────────────────────┤
│         Foundation (config, validation)           │
│  SttConfig, load_dotenv, ExitCode, exceptions    │
└─────────────────────────────────────────────────┘
```

## Граф зависимостей модулей

```
stt/cli/app.py
  ├── stt/cli/transcribe.py
  │     ├── stt/core/pipeline.py
  │     │     ├── stt/core/transcriber.py    (faster-whisper)
  │     │     ├── stt/core/diarizer.py       (pyannote)
  │     │     ├── stt/core/aligner.py
  │     │     ├── stt/core/audio.py
  │     │     └── stt/exporters/
  │     ├── stt/core/audio.py
  │     ├── stt/exceptions.py
  │     └── stt/exit_codes.py
  ├── stt/cli/batch.py
  │     ├── stt/core/batch.py
  │     │     └── stt/core/pipeline.py
  │     └── stt/exit_codes.py
  └── stt/cli/models_cmd.py
        └── stt/exit_codes.py
```

## Alignment алгоритм

Мерж Whisper-сегментов с pyannote-диаризацией: `stt/core/aligner.py`

Для каждого Whisper-сегмента:
1. Вычислить overlap с каждым diarization turn
2. Взять спикера с максимальным overlap
3. Если overlap = 0 — взять ближайший turn по времени
4. Создать новый Segment через `dataclasses.replace()` (иммутабельность)

```
overlap = max(0, min(seg_end, turn_end) - max(seg_start, turn_start))
```

## Batch-обработка

`BatchRunner` обрабатывает файлы последовательно, не останавливаясь на ошибках:
- Все успешны → `ExitCode.SUCCESS` (0)
- Частичный успех → `ExitCode.PARTIAL_SUCCESS` (10)
- Все провалились → `ExitCode.ERROR_GENERAL` (1)

`--skip-existing` проверяет наличие выходных файлов по всем запрошенным форматам.
`--recursive` сохраняет структуру поддиректорий в output.

## Конфигурация (приоритеты)

```
CLI-флаги  >  YAML-файл  >  Значения по умолчанию
```

YAML загружается из:
1. Путь, переданный напрямую
2. Переменная `STT_CONFIG`
3. `./config.yaml` (CWD fallback)

Переменные окружения загружаются автоматически из `.env` через `python-dotenv` (вызов `load_dotenv()` в `config.py`).

Приоритет `model_dir`: CLI `--model-dir` > `STT_MODEL_DIR` env > YAML > дефолт `models`

## Model Storage

Модели хранятся в директории проекта `./models/` по умолчанию. Скачивание: `stt models download large-v3 --model-dir models` (скачивает Whisper + pyannote модели).

`model_dir` протягивается через всю цепочку:
- `PipelineConfig.model_dir` → `TranscriberConfig.model_dir` → `WhisperModel(download_root=...)`
- `PipelineConfig.model_dir` → `DiarizerConfig.cache_dir` → `Pipeline.from_pretrained(cache_dir=...)`

### Совместимость с pyannote 4.x

В pyannote.audio 4.x:
- `Pipeline.from_pretrained()` принимает `token=` (не `use_auth_token=`)
- Пайплайн возвращает `DiarizeOutput` вместо `Annotation` — аннотация извлекается через `.speaker_diarization`

## Экспорт

Dispatcher `export_transcript()` принимает строку форматов через запятую (`"json,txt,srt"`):
- Один формат `json` без `output_dir` → вывод в stdout (строка)
- С `output_dir` → создание файлов (mkdir -p автоматически)
- Имя файла = stem исходного аудио + расширение формата
