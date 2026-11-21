# KoboldCpp · Haswell AVX2 and Polaris Vulkan

[English](README.md) · [Скачать](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/tag/v1.125-20261121-f1df2675) · [Документация](performance/RUN.txt) · [Данные](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261117-d9b2e6b5/results/performance-2026-11-11-kernels.json) · [Upstream](https://github.com/LostRuins/koboldcpp)

Переносимый движок Windows x64 для Intel Haswell и AMD Polaris / RX 580. Экспериментальный форк KoboldCpp с правками компиляции, ядер и NUMA для локальных GGUF-моделей текста и медиа.

**Запуск.** Распакуйте ZIP для Windows и запустите `koboldcpp-nocuda.exe`. Укажите свои веса GGUF. Для сборки O3/LTO используйте `py -3.12 performance/setup.py --build`, проверенные параметры — в [RUN.txt](performance/RUN.txt).

В сборку от 21 ноября вошла распаковка строк Q8_0 через AVX2 на Haswell. При ширине 4096 медиана времени оператора снизилась с 0,706 до 0,553 мкс. Выход совпал побайтно. [Operator data / Замеры](results/haswell-q8-0-decode-20261121.json).

В сборке Haswell от 19 ноября оператор преобразования активаций Q8_K ускорился в 3,16–3,31 раза в парных замерах. Все 1 200 проверок совпали с эталоном побайтно. [Замеры оператора](results/haswell-q8k-20261119.json).

**Проверка.** В сравнении на RX 580 от 11 ноября 2026 года ABDM Normal заняла 151,281 → 129,601 с, запрос YuE2 Vulkan — 43,729 → 40,155 с. В отчёте есть хеши файлов и повторы. В отдельном сравнении от 18 ноября с Native `ec425dc0` медианы прогретого ABDM Normal без профилировщика составили 117,04 и 116,91 с.

[November 18 Vulkan runs / Замеры Vulkan 18 ноября](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261118-ec425dc0/results/vulkan-rx580-20261118.json) · [Patch series / Патчи](performance/patches/series/) · [Licenses and corresponding source / Лицензии и исходники](performance/NOTICE.md) · [Media pipeline / Медиа](https://github.com/aktalascloud/koboldcpp-media-pipeline)

[Все проекты](https://github.com/aktalascloud)
