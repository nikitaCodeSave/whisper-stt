# API: Local STT Service

## Общие сведения

**Команда:** `stt`
**Версия:** 1.0.0

## Глобальные опции

| Флаг | Описание |
|------|----------|
| `--help, -h` | Справка |
| `--version, -V` | Версия |
| `--verbose, -v` | Подробный вывод (уровни: -v, -vv) |

## Команды

### stt transcribe

Транскрибация одного аудиофайла.

**Синтаксис:**
```bash
stt transcribe <AUDIO_FILE> [OPTIONS]
```

**Аргументы:**

| Аргумент | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `AUDIO_FILE` | Path | Да | Путь к аудиофайлу |

**Опции:**

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--model, -m` | String | `large-v3` | Модель Whisper |
| `--language, -l` | String | `ru` | Язык аудио (`ru`, `en`, `auto`) |
| `--format, -f` | String | `json` | Формат(ы) вывода: `json`, `txt`, `srt` (через запятую) |
| `--output, -o` | Path | `.` | Директория для результатов |
| `--no-diarize` | Flag | false | Отключить диаризацию |
| `--num-speakers` | Int | auto | Число спикеров (подсказка диаризатору) |
| `--min-speakers` | Int | 1 | Минимальное число спикеров |
| `--max-speakers` | Int | 8 | Максимальное число спикеров |
| `--device` | String | `cuda` | Устройство: `cuda`, `cpu` |
| `--compute-type` | String | `float16` | Тип вычислений: `float16`, `int8_float16`, `int8` |
| `--model-dir` | String | `models` | Директория хранения моделей |

**Примеры:**
```bash
# Базовая транскрипция с диаризацией
stt transcribe meeting.mp3

# Без диаризации, формат SRT
stt transcribe lecture.wav --no-diarize --format srt

# Указать число спикеров и все форматы
stt transcribe call.m4a --num-speakers 2 --format json,txt,srt

# Быстрая обработка с маленькой моделью
stt transcribe note.ogg --model small --output ./results/

# С подсказкой диапазона спикеров
stt transcribe meeting.mp3 --min-speakers 3 --max-speakers 6
```

---

### stt batch

Batch-обработка нескольких файлов.

**Синтаксис:**
```bash
stt batch <INPUT_DIR> [OPTIONS]
```

**Аргументы:**

| Аргумент | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `INPUT_DIR` | Path | Да | Директория с аудиофайлами |

**Опции:**
Все опции из `stt transcribe` плюс:

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--recursive, -r` | Flag | false | Обрабатывать поддиректории |
| `--pattern` | String | `*` | Glob-паттерн файлов (`*.mp3`) |
| `--skip-existing` | Flag | false | Пропустить уже обработанные |

**Примеры:**
```bash
# Все файлы в директории
stt batch ./recordings/ --output ./transcripts/

# Только MP3 файлы, рекурсивно
stt batch ./data/ -r --pattern "*.mp3" --format json,txt

# Пропустить уже обработанные
stt batch ./recordings/ --skip-existing
```

---

### stt models

Управление моделями.

**Синтаксис:**
```bash
stt models <SUBCOMMAND>
```

**Подкоманды:**

| Подкоманда | Описание |
|------------|----------|
| `list` | Список доступных моделей и их статус (скачана/нет) |
| `download <MODEL>` | Скачать модель заранее |
| `info <MODEL>` | Информация о модели (размер, VRAM, язык) |

**Примеры:**
```bash
# Список моделей
stt models list

# Скачать модель заранее
stt models download large-v3

# Информация о модели
stt models info large-v3-turbo
```

## Вывод в stdout

### Прогресс (stderr)
```
Processing: meeting.mp3
  [████████████████████░░░░░░░░░░] 67% | 00:04:32 / 00:06:45
  Transcription: done (23.4s)
  Diarization: in progress...
  Diarization: done (8.2s) | 3 speakers detected
  Export: meeting.json, meeting.txt
Done: meeting.mp3 (31.6s)
```

### Результат (stdout)
При `--format json` и отсутствии `--output` JSON выводится в stdout.

## Коды возврата

| Код | Константа | Описание |
|-----|-----------|----------|
| 0 | SUCCESS | Все файлы обработаны успешно |
| 1 | ERROR_GENERAL | Общая ошибка |
| 2 | ERROR_ARGS | Ошибка аргументов (неверные флаги) |
| 3 | ERROR_FILE | Файл не найден или повреждён |
| 4 | ERROR_MODEL | Модель не найдена или ошибка загрузки |
| 5 | ERROR_GPU | GPU недоступен |
| 10 | PARTIAL_SUCCESS | Batch: часть файлов обработана с ошибками |

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `STT_MODEL_DIR` | Директория хранения моделей | `models` |
| `STT_CONFIG` | Путь к файлу конфигурации | `./config.yaml` |
| `HF_TOKEN` | HuggingFace токен (для скачивания pyannote) | — |
| `CUDA_VISIBLE_DEVICES` | GPU для использования | все доступные |

## Конфигурация

**Путь:** `./config.yaml`

**Приоритет конфига:** явный path > `STT_CONFIG` env > `./config.yaml` > дефолты
**Приоритет model_dir:** CLI `--model-dir` > `STT_MODEL_DIR` env > YAML > дефолт `models`

```yaml
# Модель по умолчанию
model: large-v3

# Язык
language: ru

# Формат вывода
format: json

# Диаризация
diarization:
  enabled: true
  max_speakers: 8

# Вычисления
device: cuda
compute_type: float16

# Директории
output_dir: ./transcripts
model_dir: models
```
