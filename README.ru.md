# KoboldCpp · Haswell AVX2 / Polaris Vulkan / NUMA
[English](README.md) · [Сборка](performance/BUILD.md) · [Патчи](performance/PATCHES.md) · [Медиакомплект](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Форк [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp), только concedo_experimental. Windows64, Haswell/AVX2 и RX580 Polaris/gfx803. Запуск напрямую **koboldcpp-nocuda.exe**, моделей в пакете нет. O3+LTO, безPGO/fast-math. Монолитная линковка KCPP_COMMONIZE=0 сохраняет LTO и явныеC ABI exports.

Последний сопоставимый ручной замер: **2026-10-25**, одна модель/параметры, CPU4/Vulkan2 повтора:

| Этап, токенов/с | Исходный сток | Последний измеренный | Разница |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.8% |
| CPU output | 16.415 | 17.043 | +3.8% |
| Vulkan prompt | 411.010 | 712.810 | +73.4% |
| Vulkan output | 17.035 | 23.655 | +38.9% |

[Все исходные повторы/диапазоны, настройки и SHA](results/measurements.json). ДиапазоныCPU перекрываются. **Для установленной1.124.1/ed587d0b новой скорости нет: проверялся запуск.** Ускорение отдельной операции не равно ускорению всей генерации. HIP/ROCm не выбран из-за падения медиа; физическая dual-NUMA не измерена.

Стартовые настройки Vulkan6/4, batch/ubatch1024, gpulayers99, NUMAauto; CPU16/20, SD-энкодер20 отдельно. [Параметры](performance/RUN.txt). [Патчи/источники](performance/PATCHES.md).

Сборка одной командой: `py -3.12 performance/setup.py --build`. Оригинальные инструменты с SHA скачиваются локально; максимум4 компилятора. [Правила](AGENTS.md). Не менять модели, драйверы, BIOS, частоты и напряжения. [Лицензии](performance/NOTICE.md). Все текущие замеры — в `results/`.
