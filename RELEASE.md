Manual photo speed · 2026-11-02
- Official experimental: `SOURCE_COMMIT`; EXE SHA256: `NATIVE_SHA256`. O3/Haswell/AVX2 + LTO, no PGO.
- Current Normal736×1104, same Ray/text/VAE/LoRAs/settings: official stock→latest Native176.670→146.815s; model-ready faces/Gater pipeline184.423→154.698s. Warm2 repeats per arm; [all initial calls, return controls, SHA and limits](results/photo-latest.json).
- New gated Polaris SCALE/MOSS FFN views: incremental0.47% Native /0.37% pipeline versus prior fork,504 materializations removed.184 CPU/Vulkan cases and17 unprofiled generations retained identical RGB. [All patches/source/licenses](performance/PATCHES.md).
- Separate media lossless PNG3:4x encoding1.566→1.052s, payload+1.4%, identical RGB/text. Whole SR HTTP ranges overlap; no whole-request gain claimed. No new LLM/video benchmark or user-settings change.
