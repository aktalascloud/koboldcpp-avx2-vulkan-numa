## English

KoboldCpp 1.126 for Haswell AVX2 and Polaris Vulkan. Native SHA-256 NATIVE_SHA256, upstream SOURCE_COMMIT. The build includes 158 new upstream commits and 76 verified local patches, with O3 and LTO.

The MOSS FFN Vulkan port uses upstream strided SCALE. The local RPC transfer bounds check and Polaris dispatch guards remain in the verified series. Compilation, cold startup, the visible window and the read-only API passed. Generation and throughput were outside the validation scope.

## Русский

KoboldCpp 1.126 для Haswell AVX2 и Polaris Vulkan собран с O3 и LTO. Вошли 158 коммитов upstream и 76 проверенных локальных патчей. MOSS FFN использует strided SCALE из upstream. Проверены сборка, холодный запуск, видимое окно и GET API. Замеров генерации для этой версии нет.

[Upstream review / Обзор](results/mainstream-2026-11-26.md) · [Startup / Запуск](results/startup-4073df7f.json) · [Patch series / Патчи](performance/patches/series/README.md) · [Licenses / Лицензии](performance/NOTICE.md)
