## English

1.125 · 2026-11-18 · Native ec425dc0. The Vulkan profiler now completes the ABDM Normal request on RX 580 with the same RGB. The tested Q8_0 wide tile took 61.12 s across the matched kernels versus 58.49 s in the control. Single-operand LDS staging took 37.47/37.10 s versus 28.31 s for FlashAttention. The build keeps the faster existing kernels. Warm unprofiled request medians were 117.04 and 116.91 s. One repeated run on the old binary gave a different RGB hash.

## Русский

1.125 · 2026-11-18 · Native ec425dc0. На RX 580 исправлена работа профилировщика Vulkan на ABDM Normal. RGB совпадает побайтно. Широкая плитка Q8_0 заняла 61,12 с GPU-времени против 58,49 с на тех же формах. FlashAttention со стадированием одного операнда занял 37,47 и 37,10 с против 28,31 с. В сборке используются более быстрые прежние ядра. Медианы прогретого запроса без профилировщика составили 117,04 и 116,91 с. В одном повторе на старом EXE RGB-хеш отличался.

[Runs and hashes / Замеры и хеши](results/vulkan-rx580-20261118.json) · [Patches / Патчи](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/tree/concedo_experimental/performance/patches/series)
