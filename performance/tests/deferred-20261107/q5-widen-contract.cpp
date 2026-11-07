#include "ggml-cpu/repack.h"
#include <immintrin.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#define UNUSED GGML_UNUSED
#define GGML_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x)))
#define GGML_F32Cx8_REARRANGE_LOAD(x,m) _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(x)),m))
void probe_q5_gemv(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__)
    // Lookup table to convert signed nibbles to signed bytes
    // Shuffle masks to rearrange delta and scale values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Mask to extract nibbles from bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m256i mone  = _mm256_set1_epi8(1);

    int64_t b_nb = n / QK_K;

    const block_q5_Kx8 * b_ptr_start = (const block_q5_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one

    // Pointers to LHS blocks of block_q8_K format
    const block_q8_K * a_ptr = a_ptr_start ;

    // Take group of eight interleaved block_q5_K structures at each pass of the loop and perform dot product operation
    for (int64_t x = 0; x < nc / 8; x++) {

        // Pointers to RHS blocks
        const block_q5_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

        // Master FP accumulators
        __m256 acc_row = _mm256_setzero_ps();
        __m256 acc_min_rows = _mm256_setzero_ps();

        for (int64_t b = 0; b < nb; b++) {

            // Load and convert to FP32 scale from block_q8_K
            const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

            // Load the scale values for the 8 blocks interleaved in block_q5_Kx8
            // col_scale_f32 rearranged so as to multiply with appropriate quants
            const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
            const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

            __m256i iacc_b = _mm256_setzero_si256();
            __m256i iacc_min_b = _mm256_setzero_si256();

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)(a_ptr[b].bsums));
            __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
            q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

            // To get appropriate high bit from a byte
            __m256i hmask = mone;
            int bit = 0;

            // Load packed hbits from the 8 blocks interleaved in block_q5_Kx8
            const __m256i rhs_raw_hbit_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh));
            const __m256i rhs_raw_hbit_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh + 32));
            const __m256i rhs_raw_hbit_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh + 64));
            const __m256i rhs_raw_hbit_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh + 96));
            const __m256i rhs_raw_hbit_0123_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh + 128));
            const __m256i rhs_raw_hbit_4567_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh + 160));
            const __m256i rhs_raw_hbit_0123_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh + 192));
            const __m256i rhs_raw_hbit_4567_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qh + 224));

            // Processes two sub blocks from each q5_K in each iteration
            for (int sb = 0; sb < QK_K / 64; sb++) {

                // Load the eight block_q5_K for two sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_lbit_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
                const __m256i rhs_raw_lbit_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
                const __m256i rhs_raw_lbit_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
                const __m256i rhs_raw_lbit_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
                const __m256i rhs_raw_lbit_0123_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
                const __m256i rhs_raw_lbit_4567_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
                const __m256i rhs_raw_lbit_0123_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
                const __m256i rhs_raw_lbit_4567_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

                // 4-bit -> 8-bit
                // First sub block of the two sub blocks processed in the iteration
                const __m256i rhs_lbit_0123_00 = _mm256_and_si256(rhs_raw_lbit_0123_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                const __m256i rhs_lbit_4567_00 = _mm256_and_si256(rhs_raw_lbit_4567_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                const __m256i rhs_lbit_0123_01 = _mm256_and_si256(rhs_raw_lbit_0123_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                const __m256i rhs_lbit_4567_01 = _mm256_and_si256(rhs_raw_lbit_4567_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                const __m256i rhs_lbit_0123_02 = _mm256_and_si256(rhs_raw_lbit_0123_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                const __m256i rhs_lbit_4567_02 = _mm256_and_si256(rhs_raw_lbit_4567_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                const __m256i rhs_lbit_0123_03 = _mm256_and_si256(rhs_raw_lbit_0123_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                const __m256i rhs_lbit_4567_03 = _mm256_and_si256(rhs_raw_lbit_4567_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                // Second sub block of the two sub blocks processed in the iteration
                const __m256i rhs_lbit_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                const __m256i rhs_lbit_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                const __m256i rhs_lbit_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                const __m256i rhs_lbit_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                const __m256i rhs_lbit_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                const __m256i rhs_lbit_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                const __m256i rhs_lbit_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                const __m256i rhs_lbit_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                const __m256i rhs_hbit_0123_00 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_0, hmask), bit), 4);
                const __m256i rhs_hbit_4567_00 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_0, hmask), bit), 4);

                const __m256i rhs_hbit_0123_01 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_1, hmask), bit), 4);
                const __m256i rhs_hbit_4567_01 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_1, hmask), bit), 4);

                const __m256i rhs_hbit_0123_02 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_2, hmask), bit), 4);
                const __m256i rhs_hbit_4567_02 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_2, hmask), bit), 4);

                const __m256i rhs_hbit_0123_03 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_3, hmask), bit), 4);
                const __m256i rhs_hbit_4567_03 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_3, hmask), bit), 4);

                const __m256i rhs_vec_0123_00 = _mm256_add_epi8(rhs_lbit_0123_00, rhs_hbit_0123_00);
                const __m256i rhs_vec_4567_00 = _mm256_add_epi8(rhs_lbit_4567_00, rhs_hbit_4567_00);

                const __m256i rhs_vec_0123_01 = _mm256_add_epi8(rhs_lbit_0123_01, rhs_hbit_0123_01);
                const __m256i rhs_vec_4567_01 = _mm256_add_epi8(rhs_lbit_4567_01, rhs_hbit_4567_01);

                const __m256i rhs_vec_0123_02 = _mm256_add_epi8(rhs_lbit_0123_02, rhs_hbit_0123_02);
                const __m256i rhs_vec_4567_02 = _mm256_add_epi8(rhs_lbit_4567_02, rhs_hbit_4567_02);

                const __m256i rhs_vec_0123_03 = _mm256_add_epi8(rhs_lbit_0123_03, rhs_hbit_0123_03);
                const __m256i rhs_vec_4567_03 = _mm256_add_epi8(rhs_lbit_4567_03, rhs_hbit_4567_03);

                hmask = _mm256_slli_epi16(hmask, 1);

                const __m256i rhs_hbit_0123_10 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_0, hmask), bit + 1), 4);
                const __m256i rhs_hbit_4567_10 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_0, hmask), bit + 1), 4);

                const __m256i rhs_hbit_0123_11 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_1, hmask), bit + 1), 4);
                const __m256i rhs_hbit_4567_11 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_1, hmask), bit + 1), 4);

                const __m256i rhs_hbit_0123_12 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_2, hmask), bit + 1), 4);
                const __m256i rhs_hbit_4567_12 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_2, hmask), bit + 1), 4);

                const __m256i rhs_hbit_0123_13 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_3, hmask), bit + 1), 4);
                const __m256i rhs_hbit_4567_13 =  _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_3, hmask), bit + 1), 4);

                bit = sb * 2 + 2;

                const __m256i rhs_vec_0123_10 = _mm256_add_epi8(rhs_lbit_0123_10, rhs_hbit_0123_10);
                const __m256i rhs_vec_4567_10 = _mm256_add_epi8(rhs_lbit_4567_10, rhs_hbit_4567_10);

                const __m256i rhs_vec_0123_11 = _mm256_add_epi8(rhs_lbit_0123_11, rhs_hbit_0123_11);
                const __m256i rhs_vec_4567_11 = _mm256_add_epi8(rhs_lbit_4567_11, rhs_hbit_4567_11);

                const __m256i rhs_vec_0123_12 = _mm256_add_epi8(rhs_lbit_0123_12, rhs_hbit_0123_12);
                const __m256i rhs_vec_4567_12 = _mm256_add_epi8(rhs_lbit_4567_12, rhs_hbit_4567_12);

                const __m256i rhs_vec_0123_13 = _mm256_add_epi8(rhs_lbit_0123_13, rhs_hbit_0123_13);
                const __m256i rhs_vec_4567_13 = _mm256_add_epi8(rhs_lbit_4567_13, rhs_hbit_4567_13);

                hmask = _mm256_slli_epi16(hmask, 1);

                uint32_t utmp_0[4], utmp_1[4];

                // Scales and Mins of corresponding sub blocks from different Q5_K structures are stored together
                // The below block is for eg to extract first sub block's scales and mins from different q5_K structures for the sb loop
                memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp_0[1] & kmask1;
                utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                utmp_0[2] = uaux_0;
                utmp_0[0] &= kmask1;

                // The below block is for eg to extract second sub block's scales and mins from different q5_K structures for the sb loop
                memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                const uint32_t uaux_1 = utmp_1[1] & kmask1;
                utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                utmp_1[2] = uaux_1;
                utmp_1[0] &= kmask1;

                // Scales of first sub block in the sb loop
                const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                // Scales of second sub block in the sb loop
                __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                // Mins of first and second sub block of q5_K block are arranged side by side
                __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                // Load the two sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
                __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
                __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
                __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

                lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                // Dot product done within 32 bit lanes and accumulated in the same vector
                // First done for first sub block and thenn for second sub block in each sb
                // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                // ...........................................................................
                // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)

                __m256i iacc_0 = _mm256_setzero_si256();
                __m256i iacc_1 = _mm256_setzero_si256();

                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));

                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));

                const __m256i first_0 = _mm256_madd_epi16(iacc_0, scales_0);
                iacc_0 = _mm256_setzero_si256();
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02 ,_mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177) ,rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));

                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03 ,_mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
                iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177) ,rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));

                iacc_0 = _mm256_add_epi32(first_0, _mm256_madd_epi16(iacc_0, scales_0));

                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));

                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));

                const __m256i first_1 = _mm256_madd_epi16(iacc_1, scales_1);
                iacc_1 = _mm256_setzero_si256();
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12 ,_mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177) ,rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));

                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13 ,_mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
                iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177) ,rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));

                iacc_1 = _mm256_add_epi32(first_1, _mm256_madd_epi16(iacc_1, scales_1));

                // Accumulate the iacc value for one sb
                __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
                // Multiply-Add with corresponding mins of q5_Kx8 with bsums
                __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                q8s = _mm256_bsrli_epi128(q8s, 4);

                // Accumulate for the complete block
                iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
            }

            // Multiply-Add with scale values for the complete super block
            acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);

        }

        // Accumulated output values permuted so as to be stored in appropriate order post accumulation
        acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
        _mm256_storeu_ps(s + (x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
    }

#else
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    ggml_gemv_q5_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}
int main(){block_q5_Kx8 w{};for(auto&d:w.d)d=0x3c00;std::memset(w.scales,255,sizeof(w.scales));std::memset(w.qs,255,sizeof(w.qs));std::memset(w.qh,255,sizeof(w.qh));
 for(int act:{0,1,16,64,127,-128}){block_q8_K x{};x.d=1.f;for(auto&q:x.qs)q=act;for(auto&q:x.bsums)q=act*16;float out[8]{};probe_q5_gemv(256,out,0,&w,&x,1,8);float expected=float(256*31*63*act);int bad=0;for(float v:out)bad+=(v!=expected);std::printf("{\"activation\":%d,\"expected\":%.9g,\"actual0\":%.9g,\"wrong_columns\":%d}\n",act,expected,out[0],bad);}
}
