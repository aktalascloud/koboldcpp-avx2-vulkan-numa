## English

KoboldCpp 1.125, November 19, 2026. Native SHA-256 `NATIVE_SHA256`, base `SOURCE_COMMIT`.

Haswell uses AVX2/BMI1 to find the signed Q8_K block maximum. It chooses the same first maximum as the scalar code and keeps the original scale, rounding and sums. All 1,200 cases matched byte for byte. Median CPU operator time at width 4,096 fell from 5.35 to 1.62 µs. The release passed startup and API readiness checks.

## Русский

KoboldCpp 1.125 от 19 ноября 2026 года. AVX2/BMI1 ускоряет поиск максимума блока Q8_K на Haswell. Порядок выбора при равном модуле, масштаб, округление и суммы сохранены. Побайтно совпали все 1 200 проверок. При ширине 4 096 медиана времени оператора упала с 5,35 до 1,62 мкс. Проверены запуск и готовность API.

[Operator measurements / Замеры оператора](results/haswell-q8k-20261119.json) · [Mainstream review / Обзор upstream](results/mainstream-2026-11-19.md) · [Patch series / Серия патчей](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/tree/v1.125-20261119-efeac32e/performance/patches/series)
