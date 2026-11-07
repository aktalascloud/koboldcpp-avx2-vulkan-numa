# Release copied Windows weight pages from the working set

Keeps the file mapping and its bytes, releases complete pages in ranges the loader already marks unused, and clamps and aligns ranges without integer wrap. Working-set evidence only; no model speed claim.

[Source](https://github.com/stew675/llama-cpp-rdna-boosts/pull/269)

Validation · 2026-11-07, source for Native `5e7c8de54808cef1173cb4f0fef25ad6024b42ecefdee7e51ed2339732f69736`:the actual patched method takes a64MiB mapped-file fixture from64MiB resident to8KiB, keeps both boundary pages and preserves every byte. Invalid/sub-page/overflowing ranges are clamped safely. This is a working-set result, not a measured whole-model speed or RAM reduction.

[All repeats, flags and limits](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261107-5e7c8de5/results/operations-2026-11-07-r2.json)
