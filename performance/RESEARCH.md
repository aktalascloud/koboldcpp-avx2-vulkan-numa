# Research decisions
Reviewed 24 October 2026:100 updated KoboldCpp PRs, all 6 open PRs,20 selected active/starred forks (up to 30 PRs and 20 default-branch commits each), and 13 global GitHub searches. This is bounded discovery, not every fork/file. Existing source pins are in sources.json.

| Source | Decision for Haswell / Windows / RX580 |
|---|---|
| [GCN FA/LDS gates](https://github.com/doteyeso-ops/rx580-vulkan-fa-lds) | Adapted and measured. Enable the GCN proprietary-driver path; accept 32 KiB LDS on GCN while retaining capability checks. No FP16 spoofing or driver replacement. |
| [Grammar cache PR2434](https://github.com/LostRuins/koboldcpp/pull/2434) | Cache reject indices instead of pointers to caller-owned buffers. Reconstruct candidates from the current call. |
| [Soprano PR2557](https://github.com/LostRuins/koboldcpp/pull/2557) | Fresh-upstream build repair: include implementation from mtmd, not llama-model. No Soprano model bundled or benchmark claimed. |
| [AVX2 reduction PR30296](https://github.com/ggml-org/llama.cpp/pull/30296) | Already present in the base. Retained; not claimed as our new patch. |
| [henk717 stability fixes](https://github.com/henk717/koboldcpp/commit/874f833a54) | Relevant tokenization/logprob lifetime and locking repairs already present. |
| [F16C PR30343](https://github.com/ggml-org/llama.cpp/pull/30343) | FP16-to-FP32 lookup-table fallback is already present in this base. No duplicate patch or new speed claim. |
| [CPU Sets PR6832](https://github.com/ggml-org/llama.cpp/pull/6832) | Windows CPU Sets/groups ideas are already adapted; affinity restored after work. |
| [NUMA mirror PR27986](https://github.com/ggml-org/llama.cpp/pull/27986), [ik PR2396](https://github.com/ikawrakow/ik_llama.cpp/pull/2396) | Linux weight replication/mbind implementations. Not advertised as a Windows dual-socket optimization; physical destination testing remains necessary. |
| [P100 patch collection](https://github.com/shinbunbun/llama-cpp-p100-patches) | Reviewed host/scheduler ideas separately from CUDA kernels. Kobold has its own mark-based penalty sampler. CUDA-specific gains are not RX580 results. |
| [FSR4 gfx803](https://github.com/Schaka/fsr4-gfx803) | Linux RADV/FSR4 driver-layer work, not a drop-in Windows LLM patch. System driver untouched. |
| [HybridMoE](https://github.com/MohammadR3za/HybridMoE) | Opt-in CPU misses/GPU hot experts, adapted to SLRU and new MoE graph guards. Dense media leaves it off. |
| [vkblas](https://github.com/lakitu12/vkblas) | GCN FP32 tile ideas adapted to GGML's own LDS layout. |
| Global alternative engines | Different quantization/model/OS assumptions do not justify replacing the required Kobold backend or protected model. |

Strict-math compiler screening covers vectorization, scheduling, alignment/unrolling and Vulkan host code. Losing variants remain local. Only measured retained policy enters production. SPIR-V already uses upstream optimization except intentionally excluded shader families; unsafe optimization flags were not forced.

GPU sampling, active MoE epilogue fusion, speculative draft and Native HIP remain rejected production options. NUMA changes worker placement, not page allocation; dual-socket throughput is unmeasured.

Performance proof compares only [original stock and latest measured result](benchmarks/latest.json), with all repeated values and matching source/binary identities. No percentages from another repository are presented as our measurements.

## 25 October follow-up
All paginated PR/issue records were indexed for 23 configured sources, then relevant comments and code diffs were reviewed. Active selected forks also include issue discovery. This is an inventory plus targeted semantic review, not every comment/diff in every fork. The check runs independently of upstream HEAD changes.

- [ik PR1187](https://github.com/ikawrakow/ik_llama.cpp/pull/1187): adapted the allocation-removal idea, retaining GCC 13 double normalization, canonical RNG draws and CDF lookup. 400,500 synthetic selections and RNG states matched; seeded CPU/Vulkan API outputs matched. The isolated 40-candidate selector is 3.17x faster, not the whole model.
- [llama PR30591](https://github.com/ggml-org/llama.cpp/pull/30591): large 2D 32-bit transposed concat uses a 32x33 shared tile (4224 bytes). Embedded shader results are bit-exact; tested large operations are 3.04-4.30x faster. Small/general views keep the original shader. Qwen3.5-shaped operations may benefit; ordinary dense decode is not multiplied by this figure.
- [ik PR2107](https://github.com/ikawrakow/ik_llama.cpp/pull/2107): a fused AVX2 LUT-unpack adaptation was bit-exact in 1,000 cases but approximately 12% slower on Haswell. Rejected.
- [ik PR1262](https://github.com/ikawrakow/ik_llama.cpp/pull/1262): IQK Q8-KV FlashAttention helper, not generic Q8 weight decoding; no blind transplant.
- [ik PR2664](https://github.com/ikawrakow/ik_llama.cpp/pull/2664): IQK shifted-codebook overflow repair. Existing vanilla/tiled IQ4_XS already uses the bounded sign trick; no matching bug found.
- [llama PR30190](https://github.com/ggml-org/llama.cpp/pull/30190): coopmat1 GQA/token folding does not apply to RX580's scalar path.
- Build fix: regenerate the embedded shader object after its generated source changes; skipping the generator with Make -o must not retain old SPIR-V.

[All operation repeats and negative probe](benchmarks/operations.json). These local operation gains must not be presented as end-to-end token or photo gains. No photo/music generation was repeated.

## 28 October review
23-source paginated PR/issues inventory and21 selected-fork bounded review refreshed without errors. Relevant commits, comments and diffs were inspected; not every file/comment in all forks.

Included generic CPU REPEAT and Vulkan RMS adaptations after explicit user authorization; speed remains unmeasured. QSA CPU PR2751 targets another model/IQK path, Windows P/E affinity PR2756 targets hybrid CPUs absent on Haswell, and AVX512/Q8_R16/AMX/RDNA3-UMA-coopmat changes were not copied into Polaris. Polaris-local-ai PR7 is model-specific router/offload/ngram configuration. New source includes WebP, TTS timing and MCP startup changes. No routine inference benchmarks.
