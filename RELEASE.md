Haswell/Polaris remaining-PR update ·2026-11-07

- Official SOURCE_COMMIT;Native SHA256 NATIVE_SHA256.65 separate Native patches,7 new adaptations;O3/Haswell+LTO,no PGO.
- Normal+separate4× phase medians **188.025→155.105s**,save32.920s/17.51% vs latest official rolling1.125. Previous0519dcbf→new saves2.745s/1.74%.9 Normal/6 SR outputs have identical RGB;settings preserved. SR against previous is33ms slower and retained. [All repeats/ranges/hashes/limits](results/performance-2026-11-07-r4.json).
- Total is a sum of phase medians,not a joint public API request;model load/return/public queue/transport excluded. Fixed arm order,small warm sample. No High/new LLM tokens/s.
- [332 scoped PR dispositions](results/pr-review-2026-11-07-r4.json),including ancestry/post-image checks and explicit model/backend deferrals;not every foreign kernel manually audited. [Operator gains and rejected experiments](results/operations-2026-11-07-r5.json). No isolated gains summed into pipeline percentages.
- Native GLU/SiTU/RMS,device/shared-node/segment ownership and release CPU expert bounds. Existing Pillow13/runtime assets/licenses preserved;no model or non-commercial weights mirrored. Exact patch catalog:[Core](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/tree/concedo_experimental/performance/patches/series).
