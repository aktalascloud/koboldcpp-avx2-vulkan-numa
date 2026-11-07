Haswell / Polaris PR,security and SDCPP update · 2026-11-07

- Official SOURCE_COMMIT; Native SHA-256 NATIVE_SHA256. O3/Haswell + LTO,no PGO.
- [58 separate patches](performance/patches/series/README.md),12 additions since the last published Native: GGUF/RPC/int8,Vulkan queues,scheduler/SDCPP memory,integer LoRA and exact IQ4_NL decoding.
- [20 PR/security/P40 reviews plus8 SDCPP diffs](performance/review-2026-11-07-r4.md). Repeated fallback planning 789.67→221.76µs,operator-only. First-miss cost and all repeats retained. [Checks and limits](results/operations-2026-11-07-r4.json).
- Exact user profiles/settings,visible dated startup and GET readiness verified. [Security scope](results/security-2026-11-07.json) · [Licenses](performance/NOTICE.md).

Manual Normal+4×: **187.107→158.073s**, **29.034s / 15.52% less time**,sum of independently measured phase medians. Same settings/seed,no High,exact RGB. [All repeats,cohort identities and scope](results/performance-2026-11-07-r3.json). No new LLM token benchmark.
