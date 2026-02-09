# Discovery Log: Local STT Service

## Дата: 2026-02-09

## Собранные требования

### Сценарии использования
- Транскрибация звонков/переговоров
- Митинги и совещания

### Режим обработки
- Batch (файлы загружаются и обрабатываются)

### Целевое окружение
- Локальный сервер с GPU (NVIDIA RTX 3090/4090, 24GB VRAM)

### Критичные фичи
- Диаризация (разделение спикеров) — **основной приоритет**

### Интерфейс
- CLI-инструмент

### Объём и нагрузка
- До 1 часа аудио в день
- Смешанная длительность файлов (от коротких до длинных)

### Хранение результатов
- Файлы на диск: JSON, TXT, SRT

### Языковой режим
- Русский + английские термины вперемешку (code-switching)

### Качество входного аудио
- Хорошие микрофоны, Zoom/Meet записи

### Ключевое требование заказчика
Провести детальный сравнительный анализ реализаций STT + диаризации:
- **Pyannote** требует авторизации на HuggingFace — нужно оценить возможность полностью локальной работы
- **NeMo Toolkit** как альтернатива для диаризации
- Сравнить faster-whisper, WhisperX, оригинальный Whisper

---

## Исследование: Сравнительный анализ реализаций

### 1. STT-движки (транскрипция)

#### faster-whisper (CTranslate2)
- **Скорость:** до 4x быстрее оригинального Whisper на GPU
- **VRAM:** <8GB для large-v3 с beam_size=5
- **Модели:** Все whisper-модели (large-v3, large-v3-turbo)
- **Квантизация:** float16, int8_float16 на GPU
- **Word timestamps:** да
- **VAD:** встроенный Silero VAD для фильтрации тишины
- **Русский:** WER ~9.84% (large-v3 stock), ~6.39% (fine-tuned antony66/whisper-large-v3-russian)
- **Формат:** Python API, генератор сегментов
- **Локальность:** полностью локальный после загрузки моделей

#### WhisperX
- **Основа:** faster-whisper backend
- **Дополнительно:** wav2vec2 alignment для точных word-level timestamps
- **Диаризация:** встроенная через pyannote-audio (требует HF token)
- **VAD:** предобработка для снижения галлюцинаций
- **Русский:** та же точность, что и faster-whisper
- **Проблема:** зависимость от pyannote = зависимость от HF token
- **Известная проблема:** конфликт зависимостей pyannote-audio 3.0 и faster-whisper

#### Оригинальный OpenAI Whisper
- **Скорость:** базовая, медленнее faster-whisper в 4x
- **VRAM:** ~10GB для large-v3
- **Нет батчинга нативно**
- **Нет word-level timestamps нативно** (только segment-level)
- **Вывод:** нет смысла использовать при наличии faster-whisper

### 2. Диаризация: сравнение подходов

#### Pyannote Speaker Diarization 3.1
- **Архитектура:** PyanNet segmentation + WeSpeaker embeddings + clustering
- **Точность:** DER 11-19% на стандартных бенчмарках
- **Скорость:** ~1.5 мин на 1 час аудио (V100 GPU)
- **VRAM:** 6-8GB достаточно
- **Overlapping speech:** да
- **Установка:** `pip install pyannote.audio`

**Проблема с HuggingFace токеном:**
- Модели gated — требуют accept user agreement + HF token
- Нужно принять условия для: speaker-diarization-3.1, segmentation, embedding
- **Решение для офлайн:** можно скачать модели один раз с токеном, затем использовать локально:
  ```python
  # Скачать один раз
  git clone https://hf.co/pyannote/speaker-diarization-community-1 /path/to/local/
  # Использовать без интернета
  pipeline = Pipeline.from_pretrained('/path/to/local/pyannote-speaker-diarization-community-1')
  ```
- **Новая модель community-1:** лучше 3.1, CC-BY-4.0 лицензия, поддерживает офлайн через git clone

#### NeMo Toolkit — Cascaded Diarization (MSDD)
- **Компоненты:** MarbleNet (VAD) + TitaNet (embeddings) + MSDD (neural diarizer)
- **Мультискейл:** несколько масштабов сегментации для лучшей точности
- **Полностью локальный:** модели скачиваются с NGC, не нужен HF token
- **ASR + Diarization:** единый пайплайн `offline_diar_with_asr_infer`
- **Русский ASR в NeMo:** `stt_ru_conformer_ctc_large` — есть, но:
  - Документация рекомендует `stt_en_conformer_ctc_*` для диаризации
  - Есть открытый вопрос о совместимости ru-модели с diarization pipeline
  - Whisper для русского значительно лучше исследован
- **Проблема:** NeMo ASR models (Conformer CTC) для русского уступают Whisper по точности
- **Решение:** использовать NeMo ТОЛЬКО для диаризации, Whisper для транскрипции

#### NeMo Toolkit — Sortformer (End-to-End)
- **Архитектура:** Transformer encoder, end-to-end дарезация
- **Преимущества:** проще деплой, единая модель
- **Ограничения:** макс 4 спикера в текущей версии
- **Лицензия:** CC-BY-NC-4.0 (non-commercial!)
- **Сравнение с Pyannote:** хуже на коротких репликах, лучше на длинных монологах
- **Вывод:** для митингов (быстрая смена спикеров) Pyannote лучше

#### SpeechBrain
- **Альтернатива:** speechbrain/spkrec-ecapa-voxceleb для speaker embeddings
- **Можно использовать** вместо pyannote для extraction embeddings
- **Менее удобен** как полный pipeline

### 3. Итоговая матрица сравнения

| Критерий | faster-whisper + pyannote | WhisperX | NeMo (diarization) + faster-whisper |
|----------|--------------------------|----------|-------------------------------------|
| Скорость STT | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ (тот же faster-whisper) |
| Точность STT (русский) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ (NeMo ASR) / ⭐⭐⭐ (Whisper) |
| Точность диаризации | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Полностью локальный | ⚠️ (1 раз нужен HF token) | ⚠️ (1 раз нужен HF token) | ✅ |
| Простота интеграции | ⭐⭐⭐ | ⭐⭐⭐⭐ (всё из коробки) | ⭐⭐ (сложная конфигурация) |
| Word-level timestamps | ✅ | ✅⭐ (wav2vec2 alignment) | ⚠️ (зависит от ASR модели) |
| Overlapping speech | ✅ | ✅ | ✅ |
| Зрелость для русского | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Лицензия | MIT + CC-BY-4.0 | BSD | Apache 2.0 / CC-BY-NC |
| Сообщество | Большое | Большое | Среднее |

### 4. Рекомендация

**Основной вариант:** faster-whisper (STT) + pyannote community-1 (диаризация)
- Лучшая точность для русского
- Pyannote community-1 — можно клонировать и работать полностью офлайн
- HF token нужен однократно при первом скачивании
- Модульная архитектура: легко заменить компоненты

**Альтернативный вариант:** faster-whisper (STT) + NeMo MSDD (диаризация)
- Полностью локальный без HF
- Более сложная конфигурация
- Может потребовать больше настройки для оптимальной точности

**Стратегия:** реализовать абстракцию над диаризатором, чтобы можно было переключаться между pyannote и NeMo.
