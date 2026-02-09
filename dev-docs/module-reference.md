# Справочник модулей

Карта всех модулей с описанием экспортируемых сущностей и ключевых функций.

## stt/ — корневой пакет

### `stt/__init__.py`
- `__version__: str = "1.0.0"`

### `stt/exit_codes.py`
- `ExitCode(IntEnum)` — коды возврата CLI

| Код | Константа | Когда |
|-----|-----------|-------|
| 0 | `SUCCESS` | Все файлы обработаны |
| 1 | `ERROR_GENERAL` | Общая ошибка |
| 2 | `ERROR_ARGS` | Неверные аргументы (конфликт --num/--min/--max) |
| 3 | `ERROR_FILE` | Файл не найден, неподдерживаемый формат, пустой |
| 4 | `ERROR_MODEL` | Модель не найдена, ошибка загрузки |
| 5 | `ERROR_GPU` | GPU недоступен, OOM |
| 10 | `PARTIAL_SUCCESS` | Batch: часть файлов с ошибками |

### `stt/exceptions.py`
- `AudioValidationError(Exception)` — ошибки валидации аудио
- `GpuError(Exception)` — GPU недоступен
- `ModelError(Exception)` — ошибка загрузки модели

### `stt/data_models.py`
- `Segment(start, end, text, speaker?, confidence?)` — сегмент транскрипции
  - `.duration` property → `end - start`
  - `__post_init__`: `start > end` → `ValueError`
- `TranscriptMetadata(source_file, duration_seconds, ...)` — метаданные
  - Defaults: `model="large-v3"`, `language="ru"`, `format_version="1.0"`
  - `created_at` = `datetime.now()` автоматически
- `TranscriptResult(metadata, segments)` — полный результат
  - `.full_text` property → `" ".join(s.text for s in segments)`

### `stt/config.py`
- При импорте модуля вызывается `load_dotenv()` — автозагрузка переменных из `.env`
- `SttConfig` — dataclass конфигурации
  - `.with_overrides(**kwargs)` → новый `SttConfig` с переопределениями
- `load_config(path?)` → `SttConfig`
  - `path=None` → проверяет `STT_CONFIG` env var
  - Файл не найден → defaults
  - Поддерживает вложенный ключ `diarization: {enabled, max_speakers}`

---

## stt/core/ — ядро обработки

### `stt/core/audio.py`
- `SUPPORTED_EXTENSIONS: set[str]` = `{.mp3, .wav, .flac, .m4a, .ogg, .opus}`
- `validate_audio_file(path: Path)` → `None` или `AudioValidationError`
  - Проверяет: существование, расширение, размер > 0

### `stt/core/speaker_hints.py`
- `SpeakerHints(num_speakers?, min_speakers=1, max_speakers=8)`
- `validate_speaker_hints(hints)` → `SpeakerHints`
  - `num_speakers` задан → `min=max=num_speakers`
  - `min > max` → `ValueError`

### `stt/core/transcriber.py`
- `TranscriberConfig(model_size, device, compute_type)`
- `Transcriber(config)` — обёртка faster-whisper
  - `.load_model()` — загрузка в GPU (проверка CUDA)
  - `.transcribe(audio_path)` → `list[Segment]` (без speaker)
  - `.unload_model()` — освобождение VRAM
- `_map_confidence(avg_logprob)` → `float` [0, 1]
  - Формула: `min(max(exp(avg_logprob), 0.0), 1.0)`

### `stt/core/diarizer.py`
- `DiarizationTurn(start, end, speaker)` — один turn
- `DiarizationResult(turns, num_speakers)` — результат диаризации
- `DiarizerConfig(num_speakers?, min_speakers, max_speakers, model_name)`
- `PyannoteDiarizer(config)` — обёртка pyannote 4.x
  - `.is_available()` — `staticmethod`, проверяет importability
  - `.load_model()` → `Pipeline.from_pretrained(token=...)` (pyannote 4.x API)
  - `.diarize(audio_path)` → `DiarizationResult`
    - Обрабатывает `DiarizeOutput` (pyannote 4.x): извлекает `.speaker_diarization` → `Annotation`
  - `.unload_model()` — освобождение

### `stt/core/aligner.py`
- `_compute_overlap(seg_start, seg_end, turn_start, turn_end)` → `float`
- `align_segments(segments, diarization)` → `list[Segment]`
  - Иммутабельный: оригинальные сегменты не изменяются
  - Пустая диаризация → сегменты без speaker

### `stt/core/pipeline.py`
- `PipelineConfig(model_size, device, compute_type, language, diarization_enabled, num_speakers?, min_speakers, max_speakers, formats, output_dir)`
- `TranscriptionPipeline(config)`
  - `.run(audio_path)` → `TranscriptResult`
  - Последовательность: validate → transcribe → unload → diarize → unload → align → export

### `stt/core/batch.py`
- `discover_audio_files(input_dir, recursive?, pattern?)` → `list[Path]`
  - Фильтрует по `SUPPORTED_EXTENSIONS`
- `BatchResult(total, succeeded, failed, errors)`
  - `.exit_code` property → `ExitCode`
- `BatchRunner(pipeline_config, skip_existing?)`
  - `.run(files, output_dir, input_base?)` → `BatchResult`
  - `input_base` → сохранение структуры поддиректорий

---

## stt/exporters/ — экспорт результатов

### `stt/exporters/__init__.py`
- `export_transcript(result, formats, output_dir?)` → `str | None`
  - `formats` — строка через запятую: `"json,txt,srt"`
  - `output_dir=None` + `json` → возвращает JSON-строку (stdout)
  - Неизвестный формат → `ValueError`
  - Создаёт `output_dir` автоматически

### `stt/exporters/json_export.py`
- `export_json(result, output: IO[str])` — JSON с indent=2, ensure_ascii=False
  - `speaker`/`confidence` = None → не включаются в вывод

### `stt/exporters/txt_export.py`
- `export_txt(result, output: IO[str])`
  - Формат: `[HH:MM:SS] SPEAKER_ID: text` или `[HH:MM:SS] text`

### `stt/exporters/srt_export.py`
- `export_srt(result, output: IO[str])`
  - Формат: `HH:MM:SS,mmm --> HH:MM:SS,mmm` + `[SPEAKER_ID] text`
  - Нумерация с 1, пустая строка между записями

---

## stt/cli/ — интерфейс командной строки

### `stt/cli/app.py`
- `app = typer.Typer(name="stt")`
- Команды: `transcribe`, `batch`
- Подприложение: `models` (list, info, download)
- `--version` / `-V`, `--help`

### `stt/cli/transcribe.py`
- `transcribe_cmd(audio_file, ...)` — одиночная транскрипция
  - Валидация файла → exit 3
  - Конфликт speaker hints → exit 2
  - `GpuError` → exit 5, `ModelError` → exit 4

### `stt/cli/batch.py`
- `batch_cmd(input_dir, ...)` — batch-обработка
  - Доп. флаги: `--recursive`, `--pattern`, `--skip-existing`
  - Несуществующая директория → exit 3
  - Пустая директория → exit 0 с warning

### `stt/cli/models_cmd.py`
- `models_app = typer.Typer(name="models")`
- `list` — таблица доступных моделей
- `info <model>` — детали модели (неизвестная → exit 4)
- `download <model> [--model-dir]` — скачивает Whisper + pyannote модели
- Модели: tiny, base, small, medium, large-v3, large-v3-turbo
