# KoboldCpp: Haswell AVX2 / RX580 Polaris

[English](README.md) · [Скачать для Windows](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/releases/latest) · [Все патчи и источники](performance/PATCHES.md) · [Медиапайплайн](https://github.com/aktalascloud/koboldcpp-media-pipeline)

Форк [оригинального KoboldCpp](https://github.com/LostRuins/koboldcpp), ветка **concedo_experimental**. Windows x64, Haswell/AVX2, RX580 Polaris/gfx803, Vulkan и Windows NUMA. **O3 + LTO, без PGO.** Основные модели в пакет не включены.

Сборка28октября: включены параллельное копирование строкCPU и кэш/предзагрузка VulkanRMS по запросу пользователя. Прирост пока не измерен; прежняя таблица сохраняет исходные даты/SHA.


## Сток и последний измеренный билд

Xeon E5-2696v3 / RX580 2048SP 8GB, одинаковый Qwen3-4B IQ4_XS. Токенов/с, 2026-10-25, EXE `fb4a6ad1`.

| Этап | Исходный сток | Последний замер | Изменение |
|---|---:|---:|---:|
| CPU prompt | 39.468 | 85.153 | +115.75% |
| CPU output | 16.415 | 17.043 | +3.82% |
| Vulkan prompt | 411.010 | 712.810 | +73.43% |
| Vulkan output | 17.035 | 23.655 | +38.86% |

Prompt — ввод, Output — выдача. CPU 16/20, Vulkan 6/4; контекст 512, batch/ubatch 256, 64 выходных токена. По 4 CPU и 2 Vulkan повтора на вариант; первые холодные прогоны и все диапазоны сохранены. Диапазоны CPU пересекаются. [Полные замеры, настройки и SHA](performance/benchmarks/latest.json). Итог учитывает и прежние оптимизации.

## Что включено

- Haswell/AVX2, исправления ABI/стека Windows, Polaris matvec, FP32-плитки и пакетные отправки.
- Windows CPU Sets/groups, опциональные HybridMoE/SLRU, grammar-cache и проверка пересборки шейдеров.
- [Идея CDF sampling](https://github.com/ikawrakow/ik_llama.cpp/pull/1187): выбор из 40 кандидатов быстрее в 3,17 раза; RNG и ответы при фиксированных seed проверены.
- [Transposed concat](https://github.com/ggml-org/llama.cpp/pull/30591): проверенные крупные 32-битные операции быстрее в 3,04–4,30 раза, побитово одинаковы. Остальные формы используют прежний путь.

Это ускорения отдельных операций. [Все повторы и отклонённые пробы](performance/benchmarks/operations.json) · [Полный список патчей со ссылками](performance/PATCHES.md).

## Запуск

Распакуйте релиз и запустите **koboldcpp-nocuda.exe** напрямую, выберите GGUF в GUI или используйте:

```text
koboldcpp-nocuda.exe --model model.gguf --usevulkan 0 --gpulayers 99 --threads 6 --blasthreads 4 --batchsize 1024 --ubatchsize 1024 --numa auto
```

CPU: `--usecpu --threads 16 --blasthreads 20 --numa auto`. [Настройки и переменные среды](performance/RUN.txt) · [Сборка](performance/BUILD.md). PowerShell-лаунчер не нужен.

Сохранённый фотоконтроль 23 октября: 163,15 с сток; 138,35 с первый / 131,83 с прогретый, 576×864, исторический пайплайн с тремя LoRA. Это не новый замер MOSS. [Текущий медиапайплайн и примеры](https://github.com/aktalascloud/koboldcpp-media-pipeline).

Ежедневно 08:00 Asia/Bishkek: PR/issues независимо от изменения upstream, сборка и проверка запуска. Бенчмарки только вручную. Физическая двухсокетная машина не измерялась.

[AGPL-3.0](LICENSE.md) · Audio.cpp Apache-2.0 · [MIT](MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md) · [Лицензии зависимостей и соответствующие исходники](performance/NOTICE.md). Уведомления MIT встроенных малых декодеров сохранены. Без ключей, личных данных, логов и кэша.

<!-- native-video-ports -->

Видео Native: JPEG95, опциональный Super-TinyVAE и разделение полного энкодера/лёгкого декодера. Возможна потеря мелких деталей; [патчи и источники](performance/PATCHES.md#native-video-ports). Новых замеров выдачи токенов не было.
