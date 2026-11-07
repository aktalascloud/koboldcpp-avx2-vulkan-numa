namespace baseline {
static inline float ggml_fwht_load(const float value) {
    return value;
}

static inline float ggml_fwht_load(const ggml_fp16_t value) {
    return ggml_fp16_to_fp32(value);
}

template<typename src_t>
static void ggml_compute_forward_fwht_impl(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == (std::is_same_v<src_t, float> ? GGML_TYPE_F32 : GGML_TYPE_F16));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const int64_t n = ne10;
    GGML_ASSERT((n & (n - 1)) == 0); // must be power of 2

    const int64_t nr = ne11 * ne12 * ne13;
    const int64_t rows_per_thread = (nr + nth - 1) / nth;
    const int64_t start_row = ith * rows_per_thread;
    const int64_t end_row = MIN(start_row + rows_per_thread, nr);

    const float scale = 1.0f / sqrtf((float)n);

#if defined(GGML_SIMD)
    const GGML_F32_VEC v_minus_one = GGML_F32_VEC_SET1(-1.0f);
#endif

    for (int64_t r = start_row; r < end_row; r++) {
        const int64_t i13 = r / (ne11 * ne12);
        const int64_t i12 = (r - i13 * ne11 * ne12) / ne11;
        const int64_t i11 = r - i13 * ne11 * ne12 - i12 * ne11;

        const src_t * src_row = (const src_t *) ((const char *) src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);
        float * dst_row = (float *) ((char *) dst->data + i11 * nb1 + i12 * nb2 + i13 * nb3);

        for (int64_t j = 0; j < n; j++) {
            dst_row[j] = ggml_fwht_load(src_row[j]) * scale;
        }

        // Scalar passes
#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
        const int step = svcntw();
#else
        const int step = GGML_F32_EPR;
#endif
#else
        const int step = n;
#endif
        for (int64_t len = 1; len < step && len < n; len <<= 1) {
            for (int64_t i = 0; i < n; i += 2 * len) {
                for (int64_t j = 0; j < len; j++) {
                    float u = dst_row[i + j];
                    float v = dst_row[i + len + j];
                    dst_row[i + j] = u + v;
                    dst_row[i + len + j] = u - v;
                }
            }
        }

        // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#if defined(GGML_SIMD)
        for (int64_t len = step; len < n; len <<= 1) {
            for (int64_t i = 0; i < n; i += 2 * len) {
                for (int64_t j = 0; j < len; j += step) {
                    GGML_F32_VEC u = GGML_F32_VEC_LOAD(dst_row + i + j);
                    GGML_F32_VEC v = GGML_F32_VEC_LOAD(dst_row + i + len + j);

                    GGML_F32_VEC_STORE(dst_row + i + j,       GGML_F32_VEC_ADD(u, v));
                    GGML_F32_VEC_STORE(dst_row + i + len + j, GGML_F32_VEC_FMA(u, v, v_minus_one));
                }
            }
        }
#endif
    }
}

}
