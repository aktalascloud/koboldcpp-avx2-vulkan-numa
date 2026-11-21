## English

KoboldCpp 1.125, November 21, 2026. Native SHA-256 NATIVE_SHA256, official source SOURCE_COMMIT. This build incorporates 200 upstream commits and the verified 75-patch local series.

Haswell AVX2 Q8_0 decoding now widens eight signed bytes at once. At width 4,096, the operator median went from 0.706 to 0.553 µs. All 65,542 cases matched the reference byte for byte. MinGW embeds the same 1,599 SPIR-V modules using a lower-memory compiler input. This build passed startup and API readiness checks.

## Русский

KoboldCpp 1.125 от 21 ноября 2026 года. Перенесены 200 коммитов upstream и проверенная серия из 75 патчей. AVX2-распаковка Q8_0 на Haswell снизила медиану времени оператора при ширине 4096 с 0,706 до 0,553 мкс. Все 65 542 проверки совпали побайтно. При новой упаковке 1 599 модулей SPIR-V тоже совпали побайтно. Проверены запуск и готовность API.

[Operator timings / Замеры оператора](results/haswell-q8-0-decode-20261121.json) · [Upstream review / Обзор](results/mainstream-2026-11-21.md) · [Patch series / Патчи](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/tree/v1.125-20261121-f1df2675/performance/patches/series)
