# KoboldCpp: Haswell AVX2 / RX 580 Polaris

[English](README.md) · **[Windows release](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest)** · [Media pipeline](https://github.com/aktalascloud/koboldcpp-media-pipeline) · [Patches](performance/PATCHES.md) · [Build](performance/BUILD.md)

Форк **[оригинального KoboldCpp](https://github.com/LostRuins/koboldcpp)**, только **concedo_experimental**. Windows x64, Haswell/AVX2, AMD RX 580 Polaris/gfx803, Vulkan и NUMA. EXE запускается напрямую; модели не включены.

## Исходный сток и последний результат

Xeon E5-2696v3 / RX 580 2048SP 8 GB, одинаковый Qwen3-4B IQ4_XS. CPU 16/20, Vulkan 6/4; контекст 512, batch/ubatch 256, 64 выходных токена. Обе GPU-сборки используют один upstream-параметр graphics queue. Скорость в токенах/с; [все повторы, диапазоны, параметры и SHA](performance/benchmarks/latest.json).

| Этап | Исходный сток | Последний результат | Изменение |
|---|---:|---:|---:|
| CPU: Ввод | 39.468 | 85.153 | +115.75% |
| CPU: Выдача | 16.415 | 17.043 | +3.82% |
| Vulkan: Ввод | 411.010 | 712.810 | +73.43% |
| Vulkan: Выдача | 17.035 | 23.655 | +38.86% |

Время Native «ввод + выдача», сток/итог, без загрузки модели: CPU: **15.287 с / 9.020 с**, 41.00% меньше времени; Vulkan: **4.847 с / 3.334 с**, 31.21% меньше времени.

Замер 2026-10-25, EXE **fb4a6ad1**, исходники **e68c9b10**. Ввод и выдача считаются отдельно; проценты не складываются. Для небольших пересекающихся различий статистическая уверенность не заявляется. Все повторы, включая первый холодный контроль, сохранены; диапазоны CPU пересекаются. Плановые сборки не измеряются автоматически.

Сохранённый фотоконтроль576×864 с тремя LoRA, лицами и Gater, без SR: **сток163,15с, первый138,35с, прогретый131,83с**. Прогретый цикл короче на19,20%. Это отдельный датированный медиазамер. [Фото, глаза и полный пайплайн](https://github.com/aktalascloud/koboldcpp-media-pipeline).

## Запуск

Распакуйте релиз и запустите **koboldcpp-nocuda.exe** двойным щелчком. Выберите GGUF в штатном GUI или используйте:

```text
koboldcpp-nocuda.exe --model model.gguf --usevulkan 0 --gpulayers 99 --threads 6 --blasthreads 4 --batchsize 1024 --ubatchsize 1024 --numa auto
```

CPU: --usecpu --threads 16 --blasthreads 20 --numa auto, с пробелами между опцией и значением. Кодировщик SD использует 20 потоков отдельно. Измеренные переменные Vulkan и CPU-команды приведены в RUN.txt релиза.

## Рабочие изменения

- O3/Haswell/AVX2, обязательный LTO, **без PGO** и fast-math; подобранные флаги tiled.
- Polaris IQ4_XS rows4/WG64, пакетные submits, FP32 GCN64/K16 и проверенная настройка GCN driver/LDS.
- Повторное использование CDF при выборе токена: без лишних выделений памяти, с проверкой RNG и ответов при фиксированных seed.
- Ускоренный крупный транспонированный concat для 32-битных данных; остальные формы сохраняют исходный путь.
- Исправления Windows ABI/стека, пересборки встроенного объекта шейдеров и времени жизни grammar-cache.
- Windows NUMA CPU Sets/groups и HybridMoE/SLRU — по запросу. Expert cache выключен для dense-медиа. Двухсокетное железо физически не измерено.

[Решения по исследованным патчам](performance/RESEARCH.md). GPU sampling, MoE-фузия, draft и Native HIP в рабочую конфигурацию не включены.

Ежедневно в 08:00 Asia/Bishkek: свежие исходники, сборка, установка, проверка SHA/запуска и релиз. **Бенчмарки — только вручную.** В окне указан день сборки. Для расписания компьютер и Codex должны работать.

[AGPL-3.0](LICENSE.md), [MIT-уведомления](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md), [сторонние компоненты](performance/NOTICE.md). Модели, ключи, кэши, логи и PID не публикуются. Исходники соответствуют тегу релиза.
