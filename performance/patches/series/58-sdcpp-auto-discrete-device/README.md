# Prefer a usable discrete GPU only for automatic placement

Retains explicit backend assignments and first-device ties; integrated/shared-memory budgets cannot silently outrank an available discrete GPU. No selected-device setting is rewritten.

[Source](https://github.com/leejet/stable-diffusion.cpp/pull/2259)
