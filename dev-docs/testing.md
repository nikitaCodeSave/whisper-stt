# Тестирование

## Запуск

```bash
# Все unit-тесты
python -m pytest tests/unit/ -v --tb=short

# С coverage
python -m pytest tests/unit/ --cov=stt --cov-report=term-missing

# Конкретный модуль
python -m pytest tests/unit/test_aligner.py -v

# Integration (требует GPU или CPU + tiny модель)
python -m pytest tests/integration/ -m integration -v
```

## Структура тестов

```
tests/
  conftest.py              # Общие fixtures
  unit/
    test_smoke.py           # import, --help
    test_exit_codes.py      # ExitCode enum
    test_data_models.py     # Segment, Metadata, Result
    test_config.py          # SttConfig, load_config, load_dotenv
    test_audio_validation.py# validate_audio_file
    test_speaker_hints.py   # SpeakerHints validation
    test_transcriber.py     # Transcriber lifecycle, confidence
    test_diarizer.py        # PyannoteDiarizer, DiarizeOutput handling
    test_aligner.py         # align_segments, overlap
    test_json_export.py     # JSON export
    test_txt_export.py      # TXT export
    test_srt_export.py      # SRT export
    test_export_dispatch.py # multi-format dispatch
    test_cli_app.py         # CLI --help, --version
    test_cli_transcribe.py  # transcribe command
    test_cli_models.py      # models subcommands
    test_pipeline.py        # TranscriptionPipeline
    test_batch.py           # discover_audio_files
    test_batch_runner.py    # BatchRunner
    test_cli_batch.py       # batch command
    test_edge_cases.py      # empty file, GPU OOM
  integration/
    (test_e2e_transcribe.py)  # TODO: tiny model + CPU
  fixtures/
    (test audio files)        # TODO: записанные аудио
```

## Статистика: 220 тестов

| Файл | Тестов | Покрывает |
|------|--------|-----------|
| test_smoke.py | 2 | Импорт, --help |
| test_exit_codes.py | 10 | Все 7 значений, IntEnum |
| test_data_models.py | 17 | Segment, Metadata, Result |
| test_config.py | 15 | Defaults, YAML, env, overrides |
| test_audio_validation.py | 12 | Exists, ext, empty, valid, set |
| test_speaker_hints.py | 8 | Defaults, num→min/max, validation |
| test_transcriber.py | 11 | Confidence, lifecycle, errors |
| test_diarizer.py | 12 | Types, availability, diarize, errors |
| test_aligner.py | 10 | Overlap, alignment (7 кейсов) |
| test_json_export.py | 7 | Structure, optional fields, file |
| test_txt_export.py | 5 | Timestamp, speaker, UTF-8 |
| test_srt_export.py | 6 | Timestamp, numbering, speaker |
| test_export_dispatch.py | 5 | Single, multi, stdout, mkdir |
| test_cli_app.py | 8 | Help, version, commands |
| test_pipeline.py | 6 | Full run, VRAM, no-diarize |
| test_cli_transcribe.py | 5 | Validation, conflicts |
| test_cli_models.py | 5 | List, info, unknown |
| test_batch.py | 7 | Discover flat/recursive/pattern |
| test_batch_runner.py | 8 | Success/partial/fail, skip |
| test_cli_batch.py | 3 | Command, nonexistent, empty |
| test_edge_cases.py | 2 | Empty file, GPU OOM |
| **Всего** | **220** | |

## Стратегия мокирования

Все ML-модели мокаются в unit-тестах:

```python
# Transcriber — mock WhisperModel + torch
@patch("stt.core.transcriber.WhisperModel")
@patch("stt.core.transcriber.torch")
def test_transcribe(mock_torch, mock_whisper_cls):
    mock_torch.cuda.is_available.return_value = True
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (iter([mock_seg]), mock_info)
    mock_whisper_cls.return_value = mock_model

# Diarizer — mock Pipeline
@patch("stt.core.diarizer.Pipeline")
def test_diarize(mock_pipeline_cls):
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = mock_annotation
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

# Pipeline — mock Transcriber + PyannoteDiarizer целиком
@patch("stt.core.pipeline.PyannoteDiarizer")
@patch("stt.core.pipeline.Transcriber")
def test_pipeline(mock_transcriber_cls, mock_diarizer_cls):
    ...
```

## Линтинг

```bash
# Ruff (E, F, I, UP, B rules)
ruff check stt/ tests/

# Mypy
mypy stt/
```

## CI-рекомендации

```bash
# Минимальный CI pipeline
python -m pytest tests/unit/ -v --tb=short
ruff check stt/ tests/
mypy stt/
```

GPU-зависимые интеграционные тесты — только ручной запуск.
