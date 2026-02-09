# Local STT Service

Локальный CLI-инструмент для транскрипции аудио в текст с диаризацией спикеров. Оптимизирован для русского языка, поддерживает английские термины. Работает полностью офлайн после установки.

## Возможности

- Транскрипция речи (faster-whisper, модель large-v3)
- Диаризация спикеров (pyannote.audio 4.x)
- Экспорт в JSON, TXT, SRT
- Batch-обработка директорий
- Полностью локальная работа на GPU

## Требования

- Python 3.11+ (рекомендуется 3.12)
- NVIDIA GPU с >= 10 GB VRAM (для large-v3)
- ffmpeg (системный)
- HuggingFace токен (для скачивания моделей pyannote)

## Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd Whisper-stt

# 2. Создать виртуальное окружение
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Установить зависимости
pip install -e ".[dev]"

# 4. Настроить окружение
cp config.yaml.example config.yaml
mkdir -p models output
```

### Настройка HuggingFace токена

Для скачивания моделей pyannote необходим токен HuggingFace:

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Примите условия использования модели [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Создайте токен в [настройках](https://huggingface.co/settings/tokens)
4. Добавьте токен в `.env`:

```bash
echo "HF_TOKEN=hf_ваш_токен" > .env
```

> Токен загружается автоматически из `.env` через `python-dotenv` — ручной `source .env` не нужен.

### Скачивание моделей

```bash
stt models download large-v3 --model-dir models
```

Команда скачает:
- Whisper large-v3 (~3.1 GB)
- Pyannote speaker-diarization + вспомогательные модели (~30 MB)

## Использование

### Транскрипция одного файла

```bash
# Базовая транскрипция (JSON в stdout)
stt transcribe meeting.mp3

# С экспортом в несколько форматов
stt transcribe meeting.mp3 --format json,txt,srt --output ./results/

# Без диаризации (быстрее)
stt transcribe lecture.wav --no-diarize --format txt

# Указать точное число спикеров
stt transcribe call.m4a --num-speakers 2 --format json,txt
```

### Batch-обработка

```bash
# Все аудиофайлы в директории
stt batch ./recordings/ --output ./transcripts/

# Рекурсивно, только MP3
stt batch ./recordings/ -r --pattern "*.mp3" --format json,txt

# Пропустить уже обработанные
stt batch ./recordings/ --skip-existing --output ./transcripts/
```

### Управление моделями

```bash
# Список доступных моделей
stt models list

# Информация о модели
stt models info large-v3

# Предварительное скачивание
stt models download large-v3 --model-dir models
```

## Примеры вывода

### JSON

```json
{
  "metadata": {
    "format_version": "1.0",
    "source_file": "meeting.mp3",
    "duration_seconds": 120.5,
    "language": "ru",
    "model": "large-v3",
    "diarization": true,
    "num_speakers": 2,
    "processing_time_seconds": 15.3
  },
  "segments": [
    {
      "start": 0.0,
      "end": 4.8,
      "text": "Добрый день, начинаем совещание.",
      "speaker": "SPEAKER_00",
      "confidence": 0.95
    },
    {
      "start": 5.2,
      "end": 9.1,
      "text": "Давайте обсудим текущий sprint.",
      "speaker": "SPEAKER_01",
      "confidence": 0.92
    }
  ],
  "full_text": "Добрый день, начинаем совещание. Давайте обсудим текущий sprint."
}
```

### TXT

```
[00:00:00] SPEAKER_00: Добрый день, начинаем совещание.
[00:00:05] SPEAKER_01: Давайте обсудим текущий sprint.
```

### SRT

```
1
00:00:00,000 --> 00:00:04,800
[SPEAKER_00] Добрый день, начинаем совещание.

2
00:00:05,200 --> 00:00:09,100
[SPEAKER_01] Давайте обсудим текущий sprint.
```

## Конфигурация

Приоритет настроек: **CLI-флаги > переменные окружения > config.yaml > значения по умолчанию**.

### config.yaml

```yaml
model: large-v3          # tiny, base, small, medium, large-v3, large-v3-turbo
language: ru              # ru, en, auto
format: json              # json, txt, srt (через запятую)
device: cuda              # cuda, cpu
compute_type: float16     # float16, int8_float16, int8
output_dir: .
model_dir: models

diarization:
  enabled: true
  min_speakers: 1
  max_speakers: 8
```

### Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `HF_TOKEN` | HuggingFace токен (для pyannote) | — |
| `STT_MODEL_DIR` | Директория хранения моделей | `models` |
| `STT_CONFIG` | Путь к файлу конфигурации | `./config.yaml` |
| `CUDA_VISIBLE_DEVICES` | GPU для использования | все |

## CLI-справка

### Опции `stt transcribe`

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| `--model, -m` | Модель Whisper | `large-v3` |
| `--language, -l` | Язык аудио | `ru` |
| `--format, -f` | Формат(ы) вывода | `json` |
| `--output, -o` | Директория результатов | `.` |
| `--no-diarize` | Отключить диаризацию | `false` |
| `--num-speakers` | Число спикеров | auto |
| `--min-speakers` | Минимум спикеров | `1` |
| `--max-speakers` | Максимум спикеров | `8` |
| `--device` | Устройство | `cuda` |
| `--compute-type` | Тип вычислений | `float16` |
| `--model-dir` | Директория моделей | `models` |

### Опции `stt batch` (дополнительно)

| Флаг | Описание | По умолчанию |
|------|----------|--------------|
| `--recursive, -r` | Обрабатывать поддиректории | `false` |
| `--pattern` | Glob-паттерн файлов | `*` |
| `--skip-existing` | Пропустить обработанные | `false` |

### Коды возврата

| Код | Описание |
|-----|----------|
| 0 | Успех |
| 1 | Общая ошибка |
| 2 | Ошибка аргументов |
| 3 | Файл не найден или повреждён |
| 4 | Ошибка модели |
| 5 | GPU недоступен |
| 10 | Batch: частичный успех |

## Поддерживаемые форматы аудио

`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.opus`

## Разработка

```bash
# Линтер
ruff check stt/

# Проверка типов
mypy stt/

# Unit-тесты
python -m pytest tests/unit/ -v

# Тесты с coverage
python -m pytest tests/unit/ --cov=stt --cov-report=term-missing
```

Подробнее в [dev-docs/dev-guide.md](dev-docs/dev-guide.md).

## Технологический стек

| Компонент | Технология |
|-----------|-----------|
| STT | faster-whisper (large-v3, float16) |
| Диаризация | pyannote.audio 4.x (speaker-diarization-3.1) |
| CLI | Typer + Rich |
| Аудио | ffmpeg |
| Конфигурация | PyYAML + python-dotenv |

## Ограничения

- NVIDIA GPU обязателен для large-v3 (>= 10 GB VRAM)
- Только batch-режим, не streaming
- Спикеры обозначаются как `SPEAKER_00`, `SPEAKER_01` (не по имени)
- Максимальная рекомендуемая длительность: до 1 часа аудио
