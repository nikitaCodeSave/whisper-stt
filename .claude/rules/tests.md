---
paths:
  - "tests/**"
---

# Правила тестирования

## Мокирование ML-моделей

ALWAYS mock все ML-модели. NEVER вызывать реальные WhisperModel или pyannote Pipeline в unit-тестах.

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

**IMPORTANT:** Mock paths должны указывать на модуль-импортёр (где `import` происходит), НЕ на исходный модуль класса.

## Конвенции

- CLI тесты: `typer.testing.CliRunner` + mock pipeline
- Именование файлов: `test_{module_name}.py` в `tests/unit/`
- Запуск: `python -m pytest tests/unit/ -v --tb=short`
