# Backlog: Качество диаризации и confidence

> Анализ проблем, обнаруженных при транскрипции `dialog.mp3` (13 сек, 2 спикера).
> Дата: 2026-02-09

---

## Обнаруженные проблемы

### 1. Все сегменты — SPEAKER_00 при num_speakers=2

**Симптом:** `metadata.num_speakers = 2`, но все 4 сегмента имеют `speaker: "SPEAKER_00"`.

**Корневая причина:** Алайнер (`stt/core/aligner.py:30-45`) назначает спикера по максимальному временному перекрытию Whisper-сегмента с pyannote-turn'ом. Если перекрытия нет — берётся ближайший turn. На коротком аудио (13 сек) Whisper и pyannote генерируют сегменты с разными границами, и все Whisper-сегменты оказываются ближе к turn'ам SPEAKER_00.

**Усугубляющий фактор:** `num_speakers` в метаданных берётся напрямую из pyannote (`stt/core/pipeline.py:78`), а не из фактически назначенных спикеров.

### 2. Одинаковый confidence у всех сегментов

**Симптом:** Все 4 сегмента имеют `confidence: 0.8962220171723645`.

**Корневая причина:** Confidence вычисляется как `math.exp(avg_logprob)` (`stt/core/transcriber.py:17-19`). Метрика `avg_logprob` — среднее по токенам сегмента. Для модели large-v3 на чётком русском аудио значения `avg_logprob` попадают в узкий диапазон (-0.05...-0.20), что даёт практически одинаковый confidence для всех сегментов.

### 3. Pyannote ненадёжен на коротком аудио

Pyannote speaker-diarization-3.1 имеет известные проблемы с аудио <15 секунд:
- Аудио <5 сек часто даёт пустой результат диаризации
- Аудио 5-15 сек часто схлопывает всех спикеров в одного
- Указание `num_speakers` на коротких клипах может ухудшить результат

Источники: [pyannote-audio #886](https://github.com/pyannote/pyannote-audio/issues/886), [pyannote-audio #1567](https://github.com/pyannote/pyannote-audio/issues/1567)

---

## Подходы к улучшению alignment спикеров

### A. Word midpoint + majority vote (рекомендуется)

Включить `word_timestamps=True` в faster-whisper. Для каждого слова определить спикера по midpoint'у слова. Segment-level спикер — majority vote по словам.

- **Effort:** Низкий — faster-whisper уже поддерживает `word_timestamps`
- **Impact:** Высокий — word probabilities варьируются от 0.2 до 0.99
- **Доп. модели:** Не нужны
- **VRAM:** +0, укладывается в текущую sequential схему
- **Overhead:** ~10-15% к времени транскрипции

Используется в: [Hotovo blog](https://www.hotovo.com/blog/whisper-from-transcription-to-speaker-diarization-part-2), [Scalastic guide](https://scalastic.io/en/whisper-pyannote-ultimate-speech-transcription/)

### B. WhisperX-style forced alignment

Wav2Vec2 CTC forced alignment даёт точные word-level timestamps. Спикер назначается per-word по перекрытию с diarization turn'ами. Сегменты разбиваются по границам смены спикера.

- **Effort:** Высокий — новая модель (~300MB), новый этап pipeline
- **Impact:** Максимальный — gold standard для alignment
- **Доп. модели:** Wav2Vec2 (есть русские модели)
- **VRAM:** ~1-2GB (загружается после выгрузки Whisper)

Источник: [m-bain/whisperX](https://github.com/m-bain/whisperX), [WhisperX paper (INTERSPEECH 2023)](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.pdf)

### C. Anchor-point mapping

Линейный проход O(n+m): для каждого слова берётся anchor (start/mid/end), назначается turn, содержащий эту точку. Post-processing: majority vote по предложениям.

- **Effort:** Средний
- **Impact:** Высокий для чистого аудио
- **Ограничение:** Не работает с overlapping speech от pyannote

Источник: [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)

### D. Diarize-first, transcribe per chunk

Сначала pyannote, потом Whisper транскрибирует каждый speaker chunk отдельно. Спикер назначен by construction.

- **Effort:** Средний
- **Impact:** Средний — Whisper хуже работает на коротких изолированных фрагментах
- **Минус:** Больше GPU compute (Whisper запускается N раз)

Источник: [yinruiqing/pyannote-whisper](https://github.com/yinruiqing/pyannote-whisper)

### E. Segment splitting at speaker boundaries

Если Whisper-сегмент покрывает несколько speaker turn'ов — разбить на подсегменты по границе смены спикера (на уровне слов).

- **Effort:** Средний (требует word timestamps)
- **Impact:** Высокий для SRT/subtitle выходных форматов

Источник: [openai/whisper Discussion #264](https://github.com/openai/whisper/discussions/264)

### Рекомендация

**Спринт 1:** Подход A (word midpoint + majority vote) — минимальный effort, максимальный ROI.
**Спринт 2:** Комбинация A + E (word-level assignment + segment splitting) — полноценное решение.
**Если нужна максимальная точность:** Подход B (WhisperX forced alignment).

---

## Подходы к улучшению confidence

### A. Word-level probability aggregation (рекомендуется)

Включить `word_timestamps=True`, вычислять confidence как geometric mean от `word.probability`:

```python
log_sum = sum(math.log(max(w.probability, 1e-10)) for w in words)
segment_confidence = math.exp(log_sum / len(words))
```

- **Effort:** Низкий
- **Impact:** Высокий — word probabilities варьируются значительно (0.2-0.99)
- **Нюанс:** `word.probability` из DTW alignment, а не из decoder logits

Источник: [faster-whisper #1358](https://github.com/SYSTRAN/faster-whisper/issues/1358)

### B. Multi-metric composite score

Комбинация `avg_logprob` + `no_speech_prob` + `compression_ratio`:

```python
base = math.exp(avg_logprob)
speech_factor = 1.0 - no_speech_prob
compression_penalty = 1.0 if compression_ratio <= 2.4 else max(0.3, 2.4 / compression_ratio)
confidence = base * speech_factor * compression_penalty
```

- **Effort:** Низкий
- **Impact:** Средний — лучше ловит hallucinations и no-speech, но `avg_logprob` всё ещё узкополосный

Источник: [whisper.cpp #1059](https://github.com/ggml-org/whisper.cpp/discussions/1059), пороги из [openai/whisper transcribe.py](https://github.com/openai/whisper/blob/main/whisper/transcribe.py)

### C. Token-level softmax (gold standard)

Извлечение raw logits из decoder, softmax per-token. Требует модификации CTranslate2/faster-whisper.

- **Effort:** Очень высокий
- **Impact:** Максимальный — истинная confidence модели

Источник: [openai/whisper PR #991](https://github.com/openai/whisper/pull/991)

### Рекомендация

**Спринт 1:** Подход A (word-level) — совместим с улучшением alignment (оба требуют `word_timestamps=True`).
**Опционально:** Добавить penalties из подхода B поверх word-level confidence.

---

## Исправление num_speakers в метаданных

### Текущее поведение

```python
# stt/core/pipeline.py:78
num_speakers = diarization_result.num_speakers  # из pyannote, ДО alignment
```

### Решение

Пересчитывать `num_speakers` из фактически назначенных спикеров ПОСЛЕ alignment:

```python
# После align_segments()
actual_speakers = {seg.speaker for seg in segments if seg.speaker is not None}
num_speakers = len(actual_speakers)
```

- **Effort:** Минимальный (1 строка)
- **Impact:** Высокий — устраняет несоответствие metadata/segments

Опционально: сохранить `raw_num_speakers` от pyannote для диагностики.

---

## Guard для короткого аудио

### Проблема

Pyannote ненадёжен на аудио <15 сек. Результаты могут быть пустыми или некорректными.

### Решение

Добавить проверку длительности перед диаризацией:

- **<5 сек:** Пропустить диаризацию, предупредить пользователя
- **5-15 сек:** Предупредить о возможной неточности, не форсировать `num_speakers`
- **>15 сек:** Штатная работа

- **Effort:** Низкий
- **Impact:** Средний — предотвращает вводящие в заблуждение результаты

---

## Приоритизированный backlog

| # | Задача | Effort | Impact | Спринт |
|---|--------|--------|--------|--------|
| 1 | Пересчёт `num_speakers` из фактических сегментов | XS | High | 1 |
| 2 | Включить `word_timestamps=True` в transcriber | S | High | 1 |
| 3 | Word-level confidence (geometric mean от word.probability) | S | High | 1 |
| 4 | Word midpoint + majority vote для назначения спикеров | M | High | 1 |
| 5 | Guard: минимальная длительность для диаризации | S | Medium | 1 |
| 6 | Warning при расхождении pyannote vs post-alignment speaker count | XS | Medium | 1 |
| 7 | Segment splitting по границам смены спикера | M | High | 2 |
| 8 | Multi-metric composite confidence (no_speech + compression) | S | Medium | 2 |
| 9 | Сохранение `raw_num_speakers` в metadata для диагностики | XS | Low | 2 |
| 10 | WhisperX-style forced alignment (Wav2Vec2) | L | Very High | 3 |

### Зависимости

- Задачи 3, 4 зависят от задачи 2 (`word_timestamps=True`)
- Задача 7 зависит от задачи 4 (word-level speaker assignment)
- Задача 10 — альтернатива задачам 4+7 (другой подход к alignment)

---

## Ключевые файлы

| Файл | Что менять |
|------|-----------|
| `stt/core/transcriber.py:17-19, 60-74` | `word_timestamps=True`, новый confidence |
| `stt/core/aligner.py:20-46` | Word-level alignment, segment splitting |
| `stt/core/pipeline.py:78` | Пересчёт `num_speakers` |
| `stt/data_models.py:10-26` | Возможно: поле `words` в Segment |
| `stt/core/diarizer.py:72-91` | Guard по длительности |

## Источники

- [m-bain/whisperX](https://github.com/m-bain/whisperX) — forced alignment + word-level speaker assignment
- [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization) — anchor-point mapping
- [yinruiqing/pyannote-whisper](https://github.com/yinruiqing/pyannote-whisper) — diarize-first approach
- [faster-whisper #1358](https://github.com/SYSTRAN/faster-whisper/issues/1358) — confidence scoring discussion
- [openai/whisper #284](https://github.com/openai/whisper/discussions/284) — word confidence
- [openai/whisper #1183](https://github.com/openai/whisper/discussions/1183) — avg_logprob interpretation
- [pyannote-audio #886](https://github.com/pyannote/pyannote-audio/issues/886) — short audio issues
- [pyannote-audio #1567](https://github.com/pyannote/pyannote-audio/issues/1567) — incorrect diarization on short files
- [WhisperX paper (INTERSPEECH 2023)](https://www.isca-archive.org/interspeech_2023/bain23_interspeech.pdf)
- [Hotovo blog: Whisper diarization](https://www.hotovo.com/blog/whisper-from-transcription-to-speaker-diarization-part-2)
- [whisper.cpp #1059](https://github.com/ggml-org/whisper.cpp/discussions/1059) — confidence color-coding
- [Adopting Whisper for Confidence Estimation (arXiv, ICASSP 2025)](https://arxiv.org/abs/2502.13446)
