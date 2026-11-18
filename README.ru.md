# KoboldCpp · Haswell AVX2 / Polaris Vulkan

[English](README.md) · [Релиз Windows](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Сборка](performance/BUILD.md) · [Патчи](performance/patches/series/) · [Медиа-API](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Форк [LostRuins/KoboldCpp](https://github.com/LostRuins/koboldcpp) для Haswell AVX2 и AMD RX 580 Polaris. Ветка `concedo_experimental`. Запуск через `koboldcpp-nocuda.exe`. Веса подключаются отдельно.

**1.125 · 2026-11-18 · Native ec425dc0** · [Замеры Vulkan на RX 580](results/vulkan-rx580-20261118.json).

Сборка O3/LTO через `py -3.12 performance/setup.py --build`. Проверенные параметры приведены в [RUN.txt](performance/RUN.txt).

В сборке от 11 ноября ABDM Normal ускорилась со 151,281 до 129,601 с, а запрос YuE2 Vulkan — с 43,729 до 40,155 с. В отчёте указаны повторы, точные SHA и результаты CPU. [Замеры](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261117-d9b2e6b5/results/performance-2026-11-11-kernels.json)

[Лицензии и соответствующий исходный код](performance/NOTICE.md).
