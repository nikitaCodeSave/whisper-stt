# Local STT Service

Локальный CLI-инструмент для транскрипции аудио в текст с диаризацией спикеров, оптимизированный для русского языка.

## Контекст проекта

Сервис обрабатывает записи звонков и совещаний: принимает аудиофайл, транскрибирует речь (русский + английские термины), разделяет по спикерам, экспортирует в JSON/TXT/SRT. Работает полностью локально на GPU после установки. Batch-режим, до 1 часа аудио в день.

## Документация

| Документ | Когда читать |
|----------|--------------|
| [docs/PRD.md](docs/PRD.md) | Бизнес-цели, scope, метрики успеха |
| [docs/SPEC.md](docs/SPEC.md) | Функциональные требования, acceptance criteria |
| [docs/api.md](docs/api.md) | CLI-контракт: команды, флаги, коды возврата |
| [docs/data-formats.md](docs/data-formats.md) | JSON-schema, форматы TXT/SRT, примеры |
| [docs/decisions/](docs/decisions/) | Архитектурные решения |
| [docs/_discovery-log.md](docs/_discovery-log.md) | Исследование и сравнение STT/диаризации |

## Ключевые решения

| Решение | Обоснование | ADR |
|---------|-------------|-----|
| faster-whisper для STT | 4x быстрее оригинала, <8GB VRAM | [ADR-001](docs/decisions/ADR-001-tech-stack.md) |
| pyannote community-1 для диаризации | Лучшая точность, CC-BY-4.0, офлайн через git clone | [ADR-001](docs/decisions/ADR-001-tech-stack.md) |
| Typer для CLI | Type-safe, автодокументация, Rich-интеграция | [ADR-001](docs/decisions/ADR-001-tech-stack.md) |

## Технологический стек

- **Язык:** Python 3.11+
- **STT:** faster-whisper (модель large-v3, float16)
- **Диаризация:** pyannote.audio 3.x (community-1)
- **CLI:** Typer + Rich
- **Аудио:** ffmpeg (системный)
- **Конфигурация:** PyYAML
- **Package manager:** pip

## Ограничения

- NVIDIA GPU обязателен (≥10GB VRAM для large-v3)
- Pyannote: однократно нужен HF_TOKEN при установке (затем полностью офлайн)
- Только batch-режим, не streaming
- Спикеры обозначаются как SPEAKER_00, SPEAKER_01 (не по имени)

## Границы для AI

### ✅ DO

- Следовать CLI-контракту из [api.md](docs/api.md)
- Следовать JSON-schema из [data-formats.md](docs/data-formats.md)
- Разделять STT-движок и диаризатор как независимые модули
- Использовать type hints везде (Typer требует)
- Обрабатывать ошибки с информативными сообщениями
- Фиксировать версии зависимостей

### ❌ DO NOT

- Не отправлять данные на внешние серверы
- Не хардкодить HF_TOKEN в код
- Не смешивать логику транскрипции и диаризации в одном модуле
- Не использовать оригинальный OpenAI Whisper (только faster-whisper)
- Не использовать NeMo Sortformer (CC-BY-NC лицензия)
