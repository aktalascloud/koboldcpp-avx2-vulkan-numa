## English

KoboldCpp 1.125, built November 21, 2026. Native SHA-256 `NATIVE_SHA256`. Official source `SOURCE_COMMIT` with the verified series of 76 patches.

KoboldCpp loads SenseNova Loop, its DFlash drafter and the Qwen3-VL vision projector in one process. The Windows ZIP includes RX 580 profiles for 16K and 32K. A single 16K startup reached the API in 23.06 s with both capabilities ready and 6.88 GiB peak dedicated WDDM memory. The installed media bundle passed cold-start and API checks with the same executable.

## Русский

SenseNova Loop, парный DFlash и vision работают в одном процессе KoboldCpp. В архиве Windows есть профили 16K и 32K. На RX 580 при одном запуске 16K с тремя GGUF API стал доступен за 23,06 с, пик выделенной VRAM составил 6,88 ГиБ. Медиасервер на том же EXE прошёл холодный запуск и проверку API. Для 32K есть профиль.

[Startup receipt / Проверка запуска](results/sensenova-loop-native-startup.json) · [Profiles / Профили](performance/configs/sensenova-loop/README.md) · [Patch series / Патчи](performance/patches/series/README.md) · [Licenses / Лицензии](performance/NOTICE.md)
