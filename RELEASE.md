## English

KoboldCpp 1.126 for Haswell AVX2 and Polaris Vulkan, built with O3 and LTO. Native SHA-256 `NATIVE_SHA256`, upstream `SOURCE_COMMIT`. This build includes 107 new upstream commits and 76 verified local patches.

The update includes recurrent-state reads in place, Vulkan indexed FlashAttention bias for Inkling 2 Mini, and the merged SD adapter refactor. The local Wan video ports follow the new parameter structure. Compilation, cold startup, the visible window, and read-only APIs passed. Throughput and media generation are outside this validation scope.

## Русский

Сборка KoboldCpp 1.126 от 28 ноября для Haswell AVX2 и Polaris Vulkan с O3 и LTO. Вошли 107 коммитов upstream и 76 проверенных локальных патчей. Встроенные движки получили чтение рекуррентного состояния на месте и Vulkan FlashAttention с компактным смещением для Inkling 2 Mini. Видеоэкспорт Wan адаптирован к обновлённому SD. Проверены компиляция, холодный запуск, окно и GET API.

[Upstream review / Обзор](results/mainstream-2026-11-28.md) · [Startup / Запуск](results/startup-dce636bd.json) · [Patch series / Патчи](performance/patches/series/README.md) · [Licenses / Лицензии](performance/NOTICE.md)
