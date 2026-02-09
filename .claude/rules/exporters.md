---
paths:
  - "stt/exporters/**"
---

# Правила экспортеров

## Интерфейс

Сигнатура каждого экспортера:
```python
def export_xxx(result: TranscriptResult, output: IO[str]) -> None:
```

## Регистрация

В `stt/exporters/__init__.py`:
```python
from stt.exporters.xxx_export import export_xxx
_EXPORTERS["xxx"] = export_xxx
```

## Обработка optional fields

- `segment.speaker` может быть `None` (при `--no-diarize`)
- `segment.confidence` может быть `None`
- Всегда проверять перед использованием

## Форматы

- JSON: `indent=2`, `ensure_ascii=False` (для русского текста)
- SRT: timestamps как `HH:MM:SS,mmm`, нумерация с 1
- TXT: timestamps как `[HH:MM:SS]`
- Спецификации: [docs/data-formats.md](../../docs/data-formats.md)
