## English

KoboldCpp 1.125 for Haswell AVX2 and Polaris Vulkan. Native SHA-256 `NATIVE_SHA256`, upstream `SOURCE_COMMIT`, 77 verified patches.

SenseNova Loop prefill keeps only the logits it needs. DFlash uses one candidate at zero confidence cutoff, with a Vulkan subgroup reduction and a float-correct fallback. The released 16K and 32K profiles keep all three GGUFs available.

16K 3.54 → 6.34 tok/s, 32K 3.78 → 6.35 tok/s on the local RX 580 in paired short generations. The benchmark receipt records the full settings, answer checks and VRAM.

## Русский

Убрала лишние логиты префилла, перенесла часть `0009` для одного кандидата и редукцию Vulkan. При одинаковом коротком запросе на местной RX 580 медианы составили 16K 3,54 → 6,34 токена/с и 32K 3,78 → 6,35 токена/с. Профили 16K и 32K включены в архив.

[Measurements / Замеры](results/sensenova-loop-speed.json) · [Profiles / Профили](performance/configs/sensenova-loop/README.md) · [Patch series / Патчи](performance/patches/series/README.md) · [Licenses / Лицензии](performance/NOTICE.md)
