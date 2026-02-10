# Результаты дебага: long-audio failures + segmentation regression

Этот документ описывает изменения, реализованные по итогам исследования в [debugging_whisper.md](debugging_whisper.md). Каждый раздел соответствует рекомендации из оригинального отчёта.

---

## Контекст проблемы

Исходная проблема: сервис падал на аудио >25-30 минут с generic `except Exception`. Диагностика была невозможна из-за отсутствия гранулярной обработки ошибок и мониторинга GPU.

В ходе реализации рекомендаций возникла регрессия: переход на `BatchedInferencePipeline` как default привёл к тому, что 13-секундный диалог (ранее 4 сегмента) стал отдаваться как 1 сегмент.

---

## 1. Гранулярная обработка ошибок (Step 1)

**Проблема:** `except Exception` маскировал тип ошибки. Невозможно было отличить OOM от inference failure от unexpected error.

**Что сделано:**

### `stt/exceptions.py` — три новых исключения:

| Исключение | Назначение |
|---|---|
| `TranscriptionError` | Ошибка inference Whisper (не OOM) |
| `DiarizationError` | Ошибка inference pyannote (не OOM) |
| `CudaOomError(GpuError)` | CUDA out-of-memory, наследует `GpuError` |

Иерархия: `CudaOomError → GpuError → Exception`. Это позволяет ловить OOM отдельно от других GPU-ошибок.

### `stt/exit_codes.py` — `ERROR_OOM = 6`:

Отдельный exit code для OOM позволяет скриптам-обёрткам различать "GPU не найден" (5) и "GPU есть, но не хватило VRAM" (6).

### `stt/core/transcriber.py` — try/except в `transcribe()`:

```
torch.cuda.OutOfMemoryError → CudaOomError
RuntimeError("out of memory") → CudaOomError  # старые версии PyTorch
RuntimeError(*) → TranscriptionError
```

**Логика:** PyTorch до 2.x бросал `RuntimeError` вместо специализированного `OutOfMemoryError`. Проверка строки `"out of memory"` покрывает оба случая.

### `stt/core/diarizer.py` — аналогичный try/except в `diarize()`:

Та же цепочка, но с `DiarizationError` вместо `TranscriptionError`.

### `stt/cli/transcribe.py` — гранулярные обработчики:

```
CudaOomError    → ERROR_OOM (6) + logger.error с traceback
GpuError        → ERROR_GPU (5)
ModelError      → ERROR_MODEL (4)
TranscriptionError/DiarizationError → ERROR_GENERAL (1) + logger.error
Exception       → ERROR_GENERAL (1) + тип ошибки в сообщении
```

**Ключевое отличие от оригинала:** `CudaOomError` ловится ДО `GpuError` (т.к. наследует его). Порядок `except` критичен. Все ошибки теперь логируются с `exc_info=True`, что даёт полный traceback в лог-файле.

---

## 2. GPU-мониторинг и управление памятью (Steps 3-4)

**Проблема:** нет данных о том, сколько VRAM реально используется и сколько остаётся после unload. Невозможно диагностировать фрагментацию.

**Что сделано:**

### Новый модуль `stt/core/gpu_utils.py`:

| Функция | Назначение |
|---|---|
| `log_gpu_memory(label)` | Логирует allocated/reserved/peak MB по label |
| `cleanup_gpu_memory(label)` | `gc.collect()` + `torch.cuda.empty_cache()` + лог |
| `configure_cuda_allocator()` | Устанавливает `PYTORCH_CUDA_ALLOC_CONF` |

### `configure_cuda_allocator()` — почему именно эти параметры:

```
max_split_size_mb:128   — запрещает разбивать аллокации >128MB
expandable_segments:True — PyTorch расширяет сегменты вместо аллокации новых
```

- `max_split_size_mb:128` уменьшает фрагментацию при последовательных load/unload (Whisper ~4.7GB → unload → pyannote ~2-3GB). Без этого allocator разбивает большие блоки, и после unload остаются мелкие фрагменты, непригодные для contiguous аллокации pyannote.
- `expandable_segments:True` лучше работает при sequential model loading — наш основной паттерн.
- Функция **не перезаписывает** существующие env-переменные (merge с приоритетом пользователя).

### `cleanup_gpu_memory()` вместо голого `torch.cuda.empty_cache()`:

В оригинале `unload_model()` делал только `empty_cache()`. Этого недостаточно:
- `gc.collect()` нужен чтобы собрать Python-объекты с CUDA tensors ДО `empty_cache()`
- Без `gc.collect()` tensors удерживаются через Python refcount, и `empty_cache()` не может их освободить
- Логирование показывает allocated vs reserved — разница = фрагментация

### Pipeline — GPU logging на каждом этапе:

```
log_gpu_memory("before_transcriber_load")
load_model()
log_gpu_memory("after_transcriber_load")
transcribe()
unload_model()  # внутри cleanup_gpu_memory("transcriber_unload")
cleanup_gpu_memory("after_transcriber_unload")  # повторный cleanup

log_gpu_memory("before_diarizer_load")
load_model()
log_gpu_memory("after_diarizer_load")
...
```

Повторный `cleanup_gpu_memory()` после `unload_model()` — намеренно. `unload_model()` чистит внутренние ссылки + вызывает cleanup. Вызов на уровне pipeline — страховка: если unload бросит exception (поймано в `finally`), pipeline-уровень cleanup всё равно отработает.

---

## 3. Pipeline try/finally — защита от утечки моделей (новое)

**Проблема:** если `transcribe()` бросал exception, `unload_model()` не вызывался. Модель оставалась в GPU.

**Что сделано в `stt/core/pipeline.py`:**

```python
transcriber = Transcriber(config)
try:
    transcriber.load_model()
    segments = transcriber.transcribe(path)
finally:
    try:
        transcriber.unload_model()
    except Exception:
        logger.exception("Failed to unload transcriber")
cleanup_gpu_memory("after_transcriber_unload")
```

**Вложенный try/except в finally:** если `unload_model()` сам бросает exception (например, CUDA error при cleanup), он не маскирует оригинальную ошибку из `transcribe()`. Без этого: `TranscriptionError` → `unload_model()` fails → оригинальная ошибка потеряна, пользователь видит "Failed to unload" вместо реальной причины.

Тест `test_unload_failure_doesnt_mask_original_error` верифицирует это поведение.

---

## 4. Subprocess isolation (Step 7)

**Проблема:** PyTorch не полностью освобождает GPU при `del model + gc.collect() + empty_cache()`. Первая модель оставляет ~254MB residual. Единственный способ — убить процесс.

**Что сделано:**

### Новый модуль `stt/core/subprocess_runner.py`:

Использует `multiprocessing` со `spawn` context (не `fork` — fork небезопасен с CUDA). Каждая ML-стадия запускается в дочернем процессе:

```
Parent: pipeline.run()
  └─ spawn child: _transcribe_worker()
       load_model() → transcribe() → unload_model()
       results → Queue
     [child exits → OS полностью освобождает GPU]
  └─ spawn child: _diarize_worker()
       load_model() → diarize() → unload_model()
       results → Queue
     [child exits → OS полностью освобождает GPU]
```

**Почему `spawn`, а не `fork`:** CUDA context наследуется при fork и вызывает segfault в дочернем процессе (документировано в PyTorch). `spawn` создаёт чистый процесс.

**Обработка зависших процессов:** `terminate()` → `join(5s)` → `kill()`. Три уровня escalation: SIGTERM, ожидание, SIGKILL.

**Сериализация через Queue:** результаты передаются как dict (JSON-совместимые), `Segment` сериализуется через `dataclasses.asdict()`.

### CLI flag `--subprocess-isolation`:

Opt-in, т.к. subprocess имеет overhead (~2-5s на spawn + model load). Полезен для:
- Long audio (>20 минут) где гарантированная очистка GPU критична
- Последовательные batch-запуски где фрагментация накапливается

### Pipeline routing:

```python
if self._config.use_subprocess:
    segments = run_transcription_subprocess(asdict(config), path)
else:
    # in-process path с try/finally
```

---

## 5. Audio preprocessing — ffmpeg returncode (новое)

**Проблема:** `subprocess.run()` для ffmpeg не проверял returncode. При ошибке конвертации (битый файл, неподдерживаемый кодек) ffmpeg возвращал code 1, но pipeline продолжал с пустым/битым WAV.

**Что сделано в `stt/core/audio.py`:**

```python
if result.returncode != 0:
    stderr_msg = result.stderr.decode(errors="replace")[:200]
    raise AudioPreprocessError(
        f"ffmpeg failed (code {result.returncode}) converting {source.name}: {stderr_msg}"
    )
```

- Проверка returncode ПЕРЕД проверкой на пустой файл
- stderr обрезается до 200 символов (ffmpeg выдаёт verbose output)
- `errors="replace"` для безопасной декодировки не-UTF-8 stderr
- Temp-файл удаляется при ошибке (не оставляет мусор)

---

## 6. Batch runner — VRAM cleanup при ошибках (новое)

**Проблема:** при обработке batch, если один файл вызывал exception, GPU memory не чистилась перед следующим файлом. Фрагментация накапливалась.

**Что сделано в `stt/core/batch.py`:**

```python
needs_cleanup = False
try:
    pipeline.run(str(audio_file), output_dir=...)
except Exception as e:
    failed += 1
    needs_cleanup = True
if needs_cleanup:
    gc.collect()
    torch.cuda.empty_cache()
```

**Почему cleanup вне `except`:** как указано в debugging_whisper.md, OOM recovery code должен выполняться ВCНЕ `except` блока. Внутри `except` traceback-объект удерживает ссылки на GPU tensors.

---

## 7. BatchedInferencePipeline + fix segmentation regression (Steps 5-6)

Это основной блок изменений, реализованный в два этапа.

### Этап 1: Переход на BatchedInferencePipeline (по рекомендации Step 5)

Рекомендация из debugging_whisper.md: использовать `BatchedInferencePipeline` с VAD-based pre-segmentation.

Были добавлены:
- `BatchedInferencePipeline` в `load_model()`: создаётся поверх `WhisperModel`
- Параметры `batch_size`, `vad_filter`, `condition_on_previous_text`, `hallucination_silence_threshold` — пробрасываются через всю config chain (SttConfig → PipelineConfig → TranscriberConfig → API call)
- `vad_parameters={"min_silence_duration_ms": 500}` — VAD с паузами >500ms
- `condition_on_previous_text=False` — предотвращает hallucination cascading
- `hallucination_silence_threshold=2.0` — подавление hallucination loops на тишине

### Этап 2: Обнаружение segmentation regression

**Симптом:** 13-секундный dialog.mp3 (ранее 4 сегмента по фразам) стал давать 1 сегмент на всё аудио.

**Корневая причина:** `BatchedInferencePipeline.transcribe()` по умолчанию использует `without_timestamps=True`. Это означает, что модель НЕ генерирует timestamp-токены (`<|0.00|>`, `<|2.50|>` и т.д.). Без них decoder не может определить границы фраз внутри VAD-чанка. Весь чанк возвращается как один сегмент.

**Верификация в исходниках faster-whisper 1.2.1:**

```
BatchedInferencePipeline.transcribe()
  → TranscriptionOptions(without_timestamps=True)  # default!
    → generate_segment_batched()
      → get_prompt()  # если without_timestamps=True, не генерирует timestamp tokens
```

Параметр `without_timestamps` корректно пробрасывается через всю цепочку. Хардкодинга нет. Передача `without_timestamps=False` включает timestamp-генерацию.

**В отличие от этого**, `WhisperModel.transcribe()` (sequential) по умолчанию использует `without_timestamps=False` — поэтому оригинальный код до всех изменений работал корректно.

### Этап 3: Решение — sequential как default, batched как opt-in

**Что сделано в `stt/core/transcriber.py`:**

```python
def transcribe(self, audio_path: str) -> list[Segment]:
    if self._config.use_batched:
        segments_iter, _info = self._batched.transcribe(
            ...,
            without_timestamps=False,  # КЛЮЧЕВОЕ: включить timestamp tokens
        )
    else:
        segments_iter, _info = self._model.transcribe(...)
```

Два пути:
1. **Default (sequential):** `WhisperModel.transcribe()` — генерирует timestamp tokens, сегментация по фразам. Проверенное поведение, работало с самого начала.
2. **Opt-in (batched):** `BatchedInferencePipeline.transcribe()` с `without_timestamps=False` — batch-обработка VAD-чанков с timestamp tokens. Быстрее для long audio, но требует явного `without_timestamps=False`.

**Почему sequential как default:**
- Это поведение, которое работало до всех изменений
- BatchedInferencePipeline требует явного `without_timestamps=False`, что неочевидно и легко потерять
- Sequential проще отлаживать — один сегмент = одна фраза, нет VAD-chunking

**Почему убран `sorted()`:**
- Оба API (sequential и batched) возвращают сегменты в хронологическом порядке
- `sorted()` создавал лишнюю копию списка (O(n) memory + O(n log n) time) для сотен сегментов на длинных файлах

**Config chain для `use_batched`:**

```
CLI --batched → SttConfig.use_batched → PipelineConfig.use_batched
  → TranscriberConfig.use_batched → if/else в transcribe()
```

Также загружается из YAML:
```yaml
whisper:
  use_batched: true
```

---

## Сводка изменённых файлов

| Файл | Изменения |
|---|---|
| `stt/exceptions.py` | +3 исключения: `TranscriptionError`, `DiarizationError`, `CudaOomError` |
| `stt/exit_codes.py` | +`ERROR_OOM = 6` |
| `stt/core/gpu_utils.py` | **Новый.** `log_gpu_memory`, `cleanup_gpu_memory`, `configure_cuda_allocator` |
| `stt/core/subprocess_runner.py` | **Новый.** Subprocess isolation для transcriber и diarizer |
| `stt/core/transcriber.py` | `BatchedInferencePipeline`, `use_batched` routing, granular error handling, cleanup |
| `stt/core/diarizer.py` | Granular error handling, `cleanup_gpu_memory` вместо голого `empty_cache` |
| `stt/core/pipeline.py` | try/finally, GPU logging, subprocess routing, config passthrough |
| `stt/core/audio.py` | ffmpeg returncode check |
| `stt/core/batch.py` | VRAM cleanup при ошибках |
| `stt/config.py` | `use_batched`, `use_subprocess`, batch params в SttConfig + config chain |
| `stt/cli/transcribe.py` | `--batched`, `--subprocess-isolation`, granular error handlers |
| `stt/cli/app.py` | `configure_cuda_allocator()` при старте |

### Тесты: +53 новых тестов (220 → 273)

| Файл | Новые тесты |
|---|---|
| `tests/unit/test_transcriber.py` | Routing (sequential vs batched), `without_timestamps=False`, параметры обоих путей |
| `tests/unit/test_config.py` | `use_batched` defaults, YAML loading, pipeline config threading |
| `tests/unit/test_pipeline.py` | `use_batched` passthrough, try/finally, GPU monitoring, subprocess routing |
| `tests/unit/test_cli_transcribe.py` | `CudaOomError` → exit 6, `TranscriptionError` → exit 1, unexpected error type |
| `tests/unit/test_batch_runner.py` | VRAM cleanup on failure |
| `tests/unit/test_audio_preprocess.py` | ffmpeg returncode, stderr в сообщении |
| `tests/unit/test_exit_codes.py` | `ERROR_OOM = 6`, member count |
| `tests/unit/test_gpu_utils.py` | **Новый.** log, cleanup, CUDA allocator config |
| `tests/unit/test_subprocess_runner.py` | **Новый.** Transcription subprocess, error handling, timeout |
| `tests/unit/test_diarizer.py` | `cleanup_gpu_memory` вместо `empty_cache` |

---

## Маппинг: рекомендация → реализация

| # | Рекомендация из debugging_whisper.md | Статус | Файлы |
|---|---|---|---|
| 1 | Гранулярные exception handlers | Done | exceptions.py, exit_codes.py, transcriber.py, diarizer.py, transcribe.py |
| 2 | Проверить совместимость torch/pyannote | Manual | — (env-specific, не в коде) |
| 3 | GPU memory logging на каждом этапе | Done | gpu_utils.py, pipeline.py |
| 4 | `PYTORCH_CUDA_ALLOC_CONF` | Done | gpu_utils.py, app.py |
| 5 | `BatchedInferencePipeline` | Done (opt-in) | transcriber.py, config chain |
| 6 | `condition_on_previous_text=False`, `vad_filter=True` | Done (defaults) | transcriber.py, config.py |
| 7 | Subprocess isolation | Done (opt-in) | subprocess_runner.py, pipeline.py, transcribe.py |

---

## Оставшиеся задачи

- **Integration test** на реальном 13s dialog.mp3: проверить что sequential даёт ~4 сегмента, batched с `without_timestamps=False` тоже даёт сегменты
- **Integration test** на 30-min audio: проверить что subprocess isolation решает OOM
- **Torch version check**: добавить startup warning если pyannote 4.x + torch != 2.8.x (рекомендация #2)
