# Security policy

Use the latest accepted Windows release. This fork records scoped source/fixture checks; a build/startup check is not a full security audit.

Legacy validation maps to official990de01 and TTS index checks to bff9a7a. HTTP repetition history is bounded upstream; this fork also bounds direct C ABI callers. GGUF dimensions and RPC transfer arithmetic have separate patches. [Evidence and limits](results/security-2026-11-07.json).

The media installer obtains the original Pillow13.0.0 wheel with a pinned SHA256. Nine image controls preserve pixels; the Native EXE build environment does not bundle Pillow. Original dependency licenses remain in place.

Report upstream vulnerabilities through [LostRuins private reporting](https://github.com/LostRuins/koboldcpp/security/advisories/new). Use the fork security reporting controls for fork-specific reports when available; ordinary issues are public. Upstream advisory metadata is maintained by upstream.
