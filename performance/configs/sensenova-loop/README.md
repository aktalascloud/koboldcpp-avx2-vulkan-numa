# SenseNova Loop 9B on RX 580

Copy the released EXE, launcher and profiles next to the three matching GGUFs. Run `Start-Native.cmd` for 16K or `Start-Native.cmd 32K`. Vulkan serves localhost on 5013.

| Context | Main GPU layers | DFlash | Draft length | Decode before → after, tok/s | Prefill before → after, tok/s | Peak VRAM, GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 16K | 54 | CPU | 2 | 3.54 → 6.34 | 45.76 → 56.78 | 5.75 |
| 32K | 54 | CPU | 2 | 3.78 → 6.35 | 41.80 → 56.78 | 5.98 |

The tested code uses the same GGUFs. Greedy completions can differ with offload or draft length changes. See the release receipt for prompt lengths, repeats, control answers and memory.

[Benchmarks](../../../results/sensenova-loop-speed.json)
