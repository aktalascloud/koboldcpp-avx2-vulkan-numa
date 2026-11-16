# Preserve external consumers in segmented graph fusion checks

Carries original visited-tensor use counts into each represented segment tensor, including inputs and views. Prevents a local segment from hiding external readers and enabling unsafe fusion. Allocation/layout and model settings are unchanged.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2240)
