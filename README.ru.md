# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[English](README.md) · [Сборка](performance/BUILD.md) · [Патчи](performance/PATCHES.md) · [Медиакомплект](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Форк [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), ветка concedo_experimental. Windows x64, Haswell/AVX2 и RX 580 Polaris/gfx803. Запуск напрямую **koboldcpp-nocuda.exe**, без весов моделей в пакете. O3 + LTO, без PGO и fast-math. Монолитная линковка KCPP_COMMONIZE=0 сохраняет LTO и явный C ABI.

Цель: максимальная корректная скорость генерации на AVX2/Polaris с постоянным слиянием официальной experimental и адаптацией полезных исправлений безопасности, моделей и функций из форков. Общий медиакомплект — один переносимый KoboldCpp.exe, один runtime и единый API фото/аудио/видео/эффектов с загрузкой по требованию; настройки и модели выбирает пользователь.

**1.125 · 2026-11-11 · Native `fdc3d11d`.** Официальная база `f49df57`. [77 патчей](performance/patches/series/).

## Замеры токенов, музыки и фото · 11 ноября

| Нагрузка | Было → стало | Изменение |
|---|---:|---:|
| CPU · prompt | 72.313 → 73.305 tok/s | ≈+1.37% |
| CPU · decode | 17.251 → 17.148 tok/s | ≈-0.60% |
| RX 580 · prompt | 312.988 → 313.767 tok/s | ≈+0.25% |
| RX 580 · decode | 34.261 → 34.381 tok/s | +0.35% |
| YuE2 · 32 s | 43.900 → 43.910 s | +0.02% |
| ABDM Normal | 150.809 → 151.281 s | +0.31% |
| ABDM 4× | 2.712 → 2.639 s | ≈-2.70% |

Тот же GGUF Qwen3-4B IQ4_XS, профиль YuE2 и профиль ABDM Normal. Текст, семантические коды, MP3, PCM и RGB совпали. Символ ≈ обозначает пересечение диапазонов. Для полного запроса YuE2 по одному тёплому замеру.

[Повторы и хеши](results/performance-2026-11-11-multimodal.json) · [Операции](results/operations-2026-11-11-multimodal.json) · [Диффы PR](results/pr-review-2026-11-11-multimodal.json)

Последний сопоставимый замер токенов: **2026-10-25**, одинаковые модель и параметры, по 4 повтора CPU и 2 Vulkan:

| Этап, токенов/с | Исходный сток | Последний измеренный | Разница |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

[Все повторы, диапазоны, настройки и SHA](results/measurements.json). Диапазоны CPU перекрываются. **Замеры токенов относятся к указанной выше дате.** Исторические числа не приписываются свежему EXE. Ускорение операции не равно ускорению всей генерации. HIP/ROCm не выбран из-за падения медиа; физическая двухсокетная NUMA не измерена.

Стартовые настройки: Vulkan — потоки 6/4, batch/ubatch 1024, gpulayers 99, NUMA auto; CPU — 16/20, SD-энкодер — отдельно 20. [Параметры](performance/RUN.txt).

Сборка: `py -3.12 performance/setup.py --build`. Инструменты с проверкой SHA скачиваются локально; максимум 4 компилятора. [Правила](AGENTS.md) · [Лицензии](performance/NOTICE.md). Все замеры находятся в `results/`.
