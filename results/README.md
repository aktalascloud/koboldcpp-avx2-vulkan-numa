# Validation records

The [November 19 RX 580 ABDM comparison](rx580-abdm-20261119.json) records seven real requests with identical RGB output. The original kernels had the lowest request median, 124.05 s. Aligned Q8_0 reads took 130.69 s and Q register reuse in FlashAttention took 140.51 s.

- [Published build and executable hash](../performance/provenance.json) · [Build and recipes](../performance/BUILD.md) · [Ordered patches](../performance/patches/series/)
- [November 19 Haswell Q8_K validation](haswell-q8k-20261119.json) · [Reproducible fixture](../performance/tests/check_q8k_haswell.py) · [Upstream and engine review](mainstream-2026-11-19.md)
- [RX 580 Vulkan runs, November 18](vulkan-rx580-20261118.json) · [Stock comparison](measurements.json) · [Photo sample and its record](photo-latest.json)
- [November 11 kernel and request measurements](performance-2026-11-11-kernels.json) · [Operation checks](operations.json) · [Upstream review](mainstream-2026-11-17.json) · [Security checks](security-2026-11-07.json)

Each result belongs to the executable hash and environment recorded in that report. Kernel timings do not measure whole requests.

The November 7 test fixtures and their raw reports are preserved together at the published `v1.125-20261118-ec425dc0` tag: [fixtures][fixtures] · [r2][r2] · [r3][r3] · [r5][r5] · [r6][r6]. Check out that tag to reproduce those fixtures with their original paths.

[fixtures]: https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/tree/v1.125-20261118-ec425dc0/performance/tests
[r2]: https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261118-ec425dc0/results/operations-2026-11-07-r2.json
[r3]: https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261118-ec425dc0/results/operations-2026-11-07-r3.json
[r5]: https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261118-ec425dc0/results/operations-2026-11-07-r5.json
[r6]: https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa/blob/v1.125-20261118-ec425dc0/results/operations-2026-11-07-r6.json
