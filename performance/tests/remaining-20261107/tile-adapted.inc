template <int SUBBLK, bool HAS_MIN, int BIAS, int NK>
static void probe_micro_4x8(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                      int i0, int j0, int num_k, int slab, float * buf, int buf_stride) {
    static_assert(!HAS_MIN && BIAS != 0, "Only existing bounded act-bias formats");
    static_assert(TILED_MICRO == 16 && TILED_TILE_K == 256, "AVX-VNNI layout expects 16x256 tiles");
    constexpr int NUM_ROWS = 4;
    constexpr int NB = TILED_TILE_K / SUBBLK;
    constexpr int NS = SUBBLK / 16;
    constexpr int NG = SUBBLK / 4;
    static_assert(SUBBLK == 16 || SUBBLK == 32, "unsupported SUBBLK");
    const int nkr  = (NK > 0) ? NK : num_k;
    const int seff = (NK > 0) ? 0 : slab;
    const int qk_stride = nkr * TILED_TILE_K;
    const int qk_off = seff * TILED_TILE_K;
    const int nb_stride = NB * nkr;
    const int nb_off = seff * NB;
    const int bs_stride = (nkr == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    const int bs_off = seff * TILED_MICRO;

    __m256i sums[NUM_ROWS];
    for (int t = 0; t < NUM_ROWS; t++) {
        sums[t] = _mm256_setzero_si256();
    }

    for (int s = 0; s < NB; s++) {
        __m256i raw[NUM_ROWS];
        for (int t = 0; t < NUM_ROWS; t++) {
            raw[t] = _mm256_setzero_si256();
        }

        for (int g = 0; g < NG; g++) {
            const int kg = s * NG + g;
            const int a_off = (j0 / 8) * (8 * qk_stride)
                + qk_off + (kg % 8) * qk_stride + (kg / 8) * 32;
            const __m256i a = _mm256_load_si256((const __m256i *) &src1.q[a_off]);
            for (int t = 0; t < NUM_ROWS; t++) {
                uint32_t q4;
                memcpy(&q4, &src0.q[(i0 + t) * qk_stride + qk_off + kg * 4], sizeof(q4));
                const __m256i w = _mm256_sub_epi8(_mm256_set1_epi32((int) q4), _mm256_set1_epi8((int8_t) BIAS));
                raw[t] = _mm256_add_epi32(raw[t], _mm256_madd_epi16(_mm256_maddubs_epi16(a, w), _mm256_set1_epi16(1)));
            }
        }

        for (int t = 0; t < NUM_ROWS; t++) {
            const __m256i scale = _mm256_set1_epi32(src0.scales[(i0 + t) * nb_stride + nb_off + s]);
            sums[t] = _mm256_add_epi32(sums[t], _mm256_mullo_epi32(raw[t], scale));
        }
    }

    // Compute the min correction after the dot products to limit register pressure.
    __m256i min_sums[NUM_ROWS];
    if constexpr (HAS_MIN) {
        for (int t = 0; t < NUM_ROWS; t++) {
            min_sums[t] = _mm256_setzero_si256();
        }
        for (int s = 0; s < NB; s++) {
            __m256i bsum = _mm256_load_si256((const __m256i *) &src1.bsums[(bs_off + s * NS) * bs_stride + j0]);
            for (int u = 1; u < NS; u++) {
                bsum = _mm256_add_epi32(bsum,
                    _mm256_load_si256((const __m256i *) &src1.bsums[(bs_off + s * NS + u) * bs_stride + j0]));
            }
            for (int t = 0; t < NUM_ROWS; t++) {
                min_sums[t] = _mm256_add_epi32(min_sums[t],
                    _mm256_mullo_epi32(bsum, _mm256_set1_epi32(src0.mins[(i0 + t) * nb_stride + nb_off + s])));
            }
        }
    }

    // Preserve the AVX2 epilogue order and FMA gating.
    const __m256 d1 = _mm256_load_ps(&src1.d[seff * TILED_MICRO + j0]);
    for (int t = 0; t < NUM_ROWS; t++) {
        const int d_off = seff * TILED_MICRO + i0 + t;
        sums[t] = _mm256_sub_epi32(sums[t], _mm256_set1_epi32(src0.mins[(i0+t)*nb_stride+nb_off]));
        __m256 res = _mm256_mul_ps(_mm256_set1_ps(src0.d[d_off]), _mm256_cvtepi32_ps(sums[t]));
        if constexpr (HAS_MIN) {
            const __m256 dmin = _mm256_set1_ps(src0.dmin[d_off]);
            const __m256 mins = _mm256_cvtepi32_ps(min_sums[t]);
#if defined(__FMA__)
            res = _mm256_fnmadd_ps(dmin, mins, res);
#else
            res = _mm256_sub_ps(res, _mm256_mul_ps(dmin, mins));
#endif
        }
        float * p = &buf[(i0 + t) * buf_stride + j0];
        const __m256 previous = _mm256_loadu_ps(p);
#if defined(__FMA__)
        const __m256 updated = _mm256_fmadd_ps(res, d1, previous);
#else
        const __m256 updated = _mm256_add_ps(previous, _mm256_mul_ps(res, d1));
#endif
        _mm256_storeu_ps(p, updated);
    }
}
template <int SUBBLK, bool HAS_MIN, int BIAS, int NK>
static void probe_tile_4x8(const tiled_tile_src0 & src0, const tiled_tile_src1 & src1,
                                       int i0, int j0, int num_k, int slab, float * buf, int buf_stride) {
    for (int i = 0; i < TILED_MICRO; i += 4) {
        for (int j = 0; j < TILED_MICRO; j += 8) {
            probe_micro_4x8<SUBBLK, HAS_MIN, BIAS, NK>(src0, src1, i0 + i, j0 + j, num_k, slab, buf, buf_stride);
        }
    }
}
static void probe_pack(tiled_tile_src1 * src1, int row0, int num_k, bool bias) {
    if(!bias)return;
    const int row_stride = num_k * TILED_TILE_K;
    uint8_t * base = (uint8_t *) src1->q + row0 * row_stride;
    for (int h = 0; h < 2; h++) {
        for (int t = 0; t < row_stride / 32; t++) {
            uint8_t * block = base + h * (8 * row_stride) + t * 32;
            __m256i v[8];
            for (int r = 0; r < 8; r++) {
                v[r] = _mm256_xor_si256(_mm256_load_si256((const __m256i *) (block + r * row_stride)), _mm256_set1_epi8((int8_t) 128));
            }

            const __m256i a0 = _mm256_unpacklo_epi32(v[0], v[1]);
            const __m256i a1 = _mm256_unpackhi_epi32(v[0], v[1]);
            const __m256i a2 = _mm256_unpacklo_epi32(v[2], v[3]);
            const __m256i a3 = _mm256_unpackhi_epi32(v[2], v[3]);
            const __m256i a4 = _mm256_unpacklo_epi32(v[4], v[5]);
            const __m256i a5 = _mm256_unpackhi_epi32(v[4], v[5]);
            const __m256i a6 = _mm256_unpacklo_epi32(v[6], v[7]);
            const __m256i a7 = _mm256_unpackhi_epi32(v[6], v[7]);
            const __m256i b0 = _mm256_unpacklo_epi64(a0, a2);
            const __m256i b1 = _mm256_unpackhi_epi64(a0, a2);
            const __m256i b2 = _mm256_unpacklo_epi64(a1, a3);
            const __m256i b3 = _mm256_unpackhi_epi64(a1, a3);
            const __m256i b4 = _mm256_unpacklo_epi64(a4, a6);
            const __m256i b5 = _mm256_unpackhi_epi64(a4, a6);
            const __m256i b6 = _mm256_unpacklo_epi64(a5, a7);
            const __m256i b7 = _mm256_unpackhi_epi64(a5, a7);
            v[0] = _mm256_permute2x128_si256(b0, b4, 0x20);
            v[1] = _mm256_permute2x128_si256(b1, b5, 0x20);
            v[2] = _mm256_permute2x128_si256(b2, b6, 0x20);
            v[3] = _mm256_permute2x128_si256(b3, b7, 0x20);
            v[4] = _mm256_permute2x128_si256(b0, b4, 0x31);
            v[5] = _mm256_permute2x128_si256(b1, b5, 0x31);
            v[6] = _mm256_permute2x128_si256(b2, b6, 0x31);
            v[7] = _mm256_permute2x128_si256(b3, b7, 0x31);

            for (int r = 0; r < 8; r++) {
                _mm256_store_si256((__m256i *) (block + r * row_stride), v[r]);
            }
        }
    }
}
