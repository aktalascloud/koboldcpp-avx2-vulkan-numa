# SenseNova Loop 9B on RX 580

Copy the rebuilt `koboldcpp-nocuda.exe`, both profiles and the launcher to the folder containing the matching GGUF files. Open `Start-Native.cmd` for 16K or run `Start-Native.cmd 32K`. KoboldCpp serves its API on the loopback interface at port 5013.

The profiles load the Q4_K_M target, its paired Q4_K_M DFlash drafter and the calibrated Q4_0 Qwen3-VL projector. Both keep vision on the CPU and use the full GPU drafter, FlashAttention, Q4_0 target KV and Q8_0 draft KV. The 16K profile uses 48 target layers on the GPU. The 32K profile uses 44 for more headroom. The tensor and cache estimates are 4.41 and 4.11 GiB, respectively. Runtime memory and output quality await testing with model weights on an RX 580.

[Model source](https://huggingface.co/satellitedown/SenseNova-Loop-9B-abliterated-GGUF) · [Loop implementation](https://github.com/GuideboardLabs/llama.cpp-sensenova-loop/tree/a7f77a788893ef2cb17c82b9ac0cb1279bcd0918) · [DFlash hidden-state patch](https://huggingface.co/satellitedown/SenseNova-Loop-9B-abliterated-GGUF/tree/2644d93673251404271a0bdf22a4a4436c0d0a54/patches)
