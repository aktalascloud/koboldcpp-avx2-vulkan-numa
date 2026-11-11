# Stage dense FlashAttention K/V blocks once per Polaris workgroup

Enables existing K/V staging for the verified wave64 AMD Windows Polaris capability, wide 128-channel F16/F32 attention and sufficient LDS. The existing tile, input precision and reduction order are retained.
