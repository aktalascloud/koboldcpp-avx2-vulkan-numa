# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[English](README.md) · [Сборка](performance/BUILD.md) · [Патчи](performance/PATCHES.md) · [Медиакомплект](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Форк [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), ветка concedo_experimental. Windows x64, Haswell/AVX2 и RX 580 Polaris/gfx803. Запуск напрямую **koboldcpp-nocuda.exe**, без весов моделей в пакете. O3 + LTO, без PGO и fast-math. Монолитная линковка KCPP_COMMONIZE=0 сохраняет LTO и явный C ABI.

Цель: максимальная корректная скорость генерации на AVX2/Polaris с постоянным слиянием официальной experimental и адаптацией полезных исправлений безопасности, моделей и функций из форков. Общий медиакомплект — один переносимый KoboldCpp.exe, один runtime и единый API фото/аудио/видео/эффектов с загрузкой по требованию; настройки и модели выбирает пользователь.

Новое 2 ноября: официальные исправления расхода VRAM YuE2, движок SheetSage2, WebP-ввод и исправление grammar/reasoning budget. [Обзор и ограничения](performance/daily-review.json). Выбранные порты MOSS из [PR2576](https://github.com/LostRuins/koboldcpp/pull/2576) и отдельный обход чтения старых Vulkan-драйверов из [PR2564](https://github.com/LostRuins/koboldcpp/pull/2564) сохранены. Все патчи и ограничения — в [каталоге](performance/PATCHES.md).

Последний сопоставимый замер токенов: **2026-10-25**, одинаковые модель и параметры, по 4 повтора CPU и 2 Vulkan:

| Этап, токенов/с | Исходный сток | Последний измеренный | Разница |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

[Все повторы, диапазоны, настройки и SHA](results/measurements.json). Диапазоны CPU перекрываются. **Для текущей 1.124.2 / 6d2fedf нового замера токенов нет.** Исторические числа не приписываются свежему EXE. Ускорение операции не равно ускорению всей генерации. HIP/ROCm не выбран из-за падения медиа; физическая двухсокетная NUMA не измерена.

Стартовые настройки: Vulkan — потоки 6/4, batch/ubatch 1024, gpulayers 99, NUMA auto; CPU — 16/20, SD-энкодер — отдельно 20. [Параметры](performance/RUN.txt).

Сборка: `py -3.12 performance/setup.py --build`. Инструменты с проверкой SHA скачиваются локально; максимум 4 компилятора. [Правила](AGENTS.md) · [Лицензии](performance/NOTICE.md). Все замеры находятся в `results/`.
