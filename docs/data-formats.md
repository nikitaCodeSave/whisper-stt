# Data Formats: Local STT Service

## Версия формата
**Текущая:** 1.0
**Совместимость:** 1.0+

## Входные форматы

### Аудиофайлы

**Расширения:** .mp3, .wav, .flac, .m4a, .ogg, .opus
**MIME-types:** audio/mpeg, audio/wav, audio/flac, audio/mp4, audio/ogg, audio/opus

**Ограничения:**

| Параметр | Значение |
|----------|----------|
| Макс. размер файла | Не ограничен (batch-режим) |
| Макс. длительность | Не ограничена (обработка чанками) |
| Рекомендуемый sample rate | 16 kHz (автоматический ресемплинг) |
| Каналы | Моно или стерео (автоматический даунмикс в моно) |

**Валидация:**
- Файл существует и доступен для чтения
- Расширение из списка поддерживаемых
- Файл содержит валидный аудиопоток (проверка через ffmpeg)

---

## Выходные форматы

### JSON (основной)

**Расширение:** .json
**MIME-type:** application/json
**Назначение:** Полная структурированная транскрипция для программной обработки

**Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "format_version": { "type": "string", "description": "Версия формата" },
        "source_file": { "type": "string", "description": "Путь к исходному файлу" },
        "duration_seconds": { "type": "number", "description": "Длительность в секундах" },
        "language": { "type": "string", "description": "Определённый язык" },
        "model": { "type": "string", "description": "Использованная модель" },
        "diarization": { "type": "boolean", "description": "Была ли диаризация" },
        "num_speakers": { "type": "integer", "description": "Число обнаруженных спикеров" },
        "processing_time_seconds": { "type": "number", "description": "Время обработки" },
        "created_at": { "type": "string", "format": "date-time" }
      },
      "required": ["format_version", "source_file", "duration_seconds", "model", "created_at"]
    },
    "segments": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "start": { "type": "number", "description": "Начало сегмента в секундах" },
          "end": { "type": "number", "description": "Конец сегмента в секундах" },
          "text": { "type": "string", "description": "Транскрибированный текст" },
          "speaker": { "type": "string", "description": "ID спикера (если диаризация)" },
          "confidence": { "type": "number", "description": "Уверенность 0.0-1.0" }
        },
        "required": ["start", "end", "text"]
      }
    },
    "full_text": { "type": "string", "description": "Полный текст без таймкодов" }
  },
  "required": ["metadata", "segments", "full_text"]
}
```

**Пример:**
```json
{
  "metadata": {
    "format_version": "1.0",
    "source_file": "meeting_2026-02-09.mp3",
    "duration_seconds": 1847.3,
    "language": "ru",
    "model": "large-v3",
    "diarization": true,
    "num_speakers": 3,
    "processing_time_seconds": 542.1,
    "created_at": "2026-02-09T14:30:00Z"
  },
  "segments": [
    {
      "start": 0.0,
      "end": 4.52,
      "text": "Добрый день, коллеги. Давайте начнём наш стендап.",
      "speaker": "SPEAKER_00",
      "confidence": 0.94
    },
    {
      "start": 4.8,
      "end": 12.34,
      "text": "Привет. Я вчера закончил deployment pipeline, сейчас тестирую на staging.",
      "speaker": "SPEAKER_01",
      "confidence": 0.91
    },
    {
      "start": 12.5,
      "end": 18.1,
      "text": "Отлично. А что с performance testing? Были какие-то bottleneck?",
      "speaker": "SPEAKER_02",
      "confidence": 0.88
    }
  ],
  "full_text": "Добрый день, коллеги. Давайте начнём наш стендап. Привет. Я вчера закончил deployment pipeline, сейчас тестирую на staging. Отлично. А что с performance testing? Были какие-то bottleneck?"
}
```

---

### TXT (читаемый)

**Расширение:** .txt
**MIME-type:** text/plain
**Назначение:** Человекочитаемая транскрипция для быстрого просмотра

**Формат:**
```
[HH:MM:SS] SPEAKER_ID: текст сегмента
```

**Пример:**
```
[00:00:00] SPEAKER_00: Добрый день, коллеги. Давайте начнём наш стендап.
[00:00:04] SPEAKER_01: Привет. Я вчера закончил deployment pipeline, сейчас тестирую на staging.
[00:00:12] SPEAKER_02: Отлично. А что с performance testing? Были какие-то bottleneck?
```

**Без диаризации:**
```
[00:00:00] Добрый день, коллеги. Давайте начнём наш стендап.
[00:00:04] Привет. Я вчера закончил deployment pipeline, сейчас тестирую на staging.
```

---

### SRT (субтитры)

**Расширение:** .srt
**MIME-type:** application/x-subrip
**Назначение:** Формат субтитров для видеоплееров

**Формат:**
```
НОМЕР
HH:MM:SS,mmm --> HH:MM:SS,mmm
[SPEAKER_ID] текст сегмента
```

**Пример:**
```
1
00:00:00,000 --> 00:00:04,520
[SPEAKER_00] Добрый день, коллеги. Давайте начнём наш стендап.

2
00:00:04,800 --> 00:00:12,340
[SPEAKER_01] Привет. Я вчера закончил deployment pipeline, сейчас тестирую на staging.

3
00:00:12,500 --> 00:00:18,100
[SPEAKER_02] Отлично. А что с performance testing? Были какие-то bottleneck?
```

---

## Общие типы данных

### Timestamp
```json
{ "type": "number", "minimum": 0.0, "description": "Время в секундах с точностью до сотых" }
```

### SpeakerID
```json
{ "type": "string", "pattern": "^SPEAKER_\\d{2}$", "description": "Идентификатор спикера: SPEAKER_00 .. SPEAKER_99" }
```

### Confidence
```json
{ "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Уверенность модели в результате" }
```

## Именование выходных файлов

Имя выходного файла = имя входного файла + расширение формата:

| Входной файл | JSON | TXT | SRT |
|-------------|------|-----|-----|
| `meeting.mp3` | `meeting.json` | `meeting.txt` | `meeting.srt` |
| `call_2024.m4a` | `call_2024.json` | `call_2024.txt` | `call_2024.srt` |

При batch-обработке структура директорий сохраняется в output-директории.
