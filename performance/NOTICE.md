# Third-party notices

KoboldCpp: LostRuins/Concedo and contributors, AGPL-3.0. GGML/llama.cpp/stable-diffusion.cpp retain upstream MIT notices and author credits. Full original source/license files remain in this fork.

HybridMoE implementation: [MohammadR3za/llama.cpp-hybrid-moe](https://github.com/MohammadR3za/llama.cpp-hybrid-moe), commit eed95add (MIT; retained under licenses/HybridMoE-MIT.txt). The HybridMoE wrapper scripts are not redistributed. Additional ideas for prefetch from ponyol/llama.cpp.patch.speed and cubetitled-ui/llama.cpp. Vulkan geometry was adapted to GGML from ideas investigated in lakitu12/vkblas; no vkblas runtime is linked. Additional inspiration/decisions: sources.json and PATCHES.md. No endorsement is claimed.

The Windows executable also packages upstream aria2c 1.37.0 (GPL-2.0-or-later), simplecpuinfo (source in kcpp_src), Python, PyInstaller bootloader (GPL exception), CustomTkinter/darkdetect, Jinja2/MarkupSafe and psutil. Release notices retain actual installed dependency licenses. An official aria2 source archive and the exact Kobold fork source accompany the binary release. Original compiler runtime exceptions apply to statically linked GCC/libstdc++ runtime components. No third-party license is replaced by this fork's AGPL label.

Embedded upstream tiny autoencoders (TAESD/TAEQI): Ollin Boer Bohan / madebyollin, MIT; license retained in licenses/TAESD-MIT.txt. TAEQI2.1 model card: https://huggingface.co/madebyollin/taeqi2_1 (MIT). The bundled decoder retains its original MIT notice.

New ports: the transposed-concat shader is adapted from [llama.cpp PR30591](https://github.com/ggml-org/llama.cpp/pull/30591), under MIT. The sampling CDF implementation independently adapts the idea in [ik_llama PR1187](https://github.com/ikawrakow/ik_llama.cpp/pull/1187), retaining GCC 13 RNG behavior. Full upstream copyright/permission notices are in licenses/llama.cpp-MIT.txt and licenses/ik_llama.cpp-MIT.txt; source and license fingerprints are in licenses/patch-sources.json.

Official experimental59ab92c adds the selective [Audio.cpp](https://github.com/0xShug0/audio.cpp) integration (Copyright2026 ShugoAI LLC, Apache-2.0) and cJSON (MIT). Full notices and the relocated native-component licenses are included under licenses/; exact source paths/commit/hashes are in licenses/upstream-native.json. No audio model weights are distributed.
