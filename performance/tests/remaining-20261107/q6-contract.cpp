#include "ggml-cpu/repack.h"
#include <immintrin.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#define UNUSED GGML_UNUSED
#define GGML_F32Cx8_LOAD(x) _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x)))
#define GGML_F32Cx8_REARRANGE_LOAD(x,m) _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(x)),m))
void probe_q6_gemv(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

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
    // Shuffle masks to rearrange delta values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Mask to extract 2 bits from packed bytes
    const __m256i m3b = _mm256_set1_epi8(3);
    // Mask to extract nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0xF);
    // Vector with each byte value 32 - Used as an subtract offset for 6 bit quantized values
    const __m256i m32s = _mm256_set1_epi8(32);

    //Mask to get appropriate scales
    __m128i scalemask1 = _mm_set_epi8(7,7,3,3,6,6,2,2,5,5,1,1,4,4,0,0);
    __m128i scalemask2 = _mm_set_epi8(15,15,11,11,14,14,10,10,13,13,9,9,12,12,8,8);

    int64_t b_nb = n / QK_K;

    const block_q6_Kx8 * b_ptr_start = (const block_q6_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q6_K structures at each pass of the loop and perform dot product operation
        for(int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q6_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_row = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 delta from block_q8_K
                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

                // Load the delta values for the 8 blocks interleaved in block_q6_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);

                __m256i iacc_b = _mm256_setzero_si256();

                // Processes eight sub blocks from each Q6_K in each iteration
                for(int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the high bits(bit 5, 6) of eight block_q6_K for eight sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_hbit_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + sb * 256));
                    const __m256i rhs_raw_hbit_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + 32 + sb * 256));
                    const __m256i rhs_raw_hbit_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + 64 + sb * 256));
                    const __m256i rhs_raw_hbit_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + 96 + sb * 256));
                    const __m256i rhs_raw_hbit_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + 128 + sb * 256));
                    const __m256i rhs_raw_hbit_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + 160 + sb * 256));
                    const __m256i rhs_raw_hbit_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + 192 + sb * 256));
                    const __m256i rhs_raw_hbit_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qh + 224 + sb * 256));

                    // 2-bit -> 8-bit
                    // hbit Values of the 0th,2nd,4th,6th sub blocks of eight block_q6_K structures for the sb loop
                    const __m256i rhs_hbit_0123_00 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_0, m3b), 4); //B00(0-7) B01(0-7) B02(0-7) B03(0-7)
                    const __m256i rhs_hbit_0123_20 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_0, 2), m3b), 4); //B20(0-7) B21(0-7) B22(0-7) B23(0-7)
                    const __m256i rhs_hbit_0123_40 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_0, 4), m3b), 4); //B40(0-7) B41(0-7) B42(0-7) B43(0-7)
                    const __m256i rhs_hbit_0123_60 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_0, 6), m3b), 4); //B60(0-7) B61(0-7) B62(0-7) B63(0-7)

                    const __m256i rhs_hbit_4567_00 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_0, m3b), 4); //B04(0-7) B05(0-7) B06(0-7) B07(0-7)
                    const __m256i rhs_hbit_4567_20 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_0, 2), m3b), 4); //B24(0-7) B25(0-7) B26(0-7) B27(0-7)
                    const __m256i rhs_hbit_4567_40 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_0, 4), m3b), 4); //B44(0-7) B45(0-7) B46(0-7) B47(0-7)
                    const __m256i rhs_hbit_4567_60 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_0, 6), m3b), 4); //B64(0-7) B65(0-7) B66(0-7) B67(0-7)

                    const __m256i rhs_hbit_0123_01 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_1, m3b), 4); //B00(8-15) B01(8-15) B02(8-15) B03(8-15)
                    const __m256i rhs_hbit_0123_21 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_1, 2), m3b), 4); //B20(8-15) B21(8-15) B22(8-15) B23(8-15)
                    const __m256i rhs_hbit_0123_41 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_1, 4), m3b), 4); //B40(8-15) B41(8-15) B42(8-15) B43(8-15)
                    const __m256i rhs_hbit_0123_61 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_1, 6), m3b), 4); //B60(8-15) B61(8-15) B62(8-15) B63(8-15)

                    const __m256i rhs_hbit_4567_01 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_1, m3b), 4); //B04(8-15) B05(8-15) B06(8-15) B07(8-15)
                    const __m256i rhs_hbit_4567_21 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_1, 2), m3b), 4); //B24(8-15) B25(8-15) B26(8-15) B27(8-15)
                    const __m256i rhs_hbit_4567_41 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_1, 4), m3b), 4); //B44(8-15) B45(8-15) B46(8-15) B47(8-15)
                    const __m256i rhs_hbit_4567_61 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_1, 6), m3b), 4); //B64(8-15) B65(8-15) B66(8-15) B67(8-15)

                    // hbit Values of the 1st,3rd,5th,7th sub blocks of eight block_q6_K structures for the sb loop
                    const __m256i rhs_hbit_0123_10 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_2, m3b), 4); //B10(0-7) B11(0-7) B12(0-7) B13(0-7)
                    const __m256i rhs_hbit_0123_30 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_2, 2), m3b), 4); //B30(0-7) B31(0-7) B32(0-7) B33(0-7)
                    const __m256i rhs_hbit_0123_50 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_2, 4), m3b), 4); //B50(0-7) B51(0-7) B52(0-7) B53(0-7)
                    const __m256i rhs_hbit_0123_70 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_2, 6), m3b), 4); //B70(0-7) B71(0-7) B72(0-7) B73(0-7)

                    const __m256i rhs_hbit_4567_10 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_2, m3b), 4); //B14(0-7) B15(0-7) B16(0-7) B17(0-7)
                    const __m256i rhs_hbit_4567_30 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_2, 2), m3b), 4); //B34(0-7) B35(0-7) B36(0-7) B37(0-7)
                    const __m256i rhs_hbit_4567_50 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_2, 4), m3b), 4); //B54(0-7) B55(0-7) B56(0-7) B57(0-7)
                    const __m256i rhs_hbit_4567_70 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_2, 6), m3b), 4); //B74(0-7) B75(0-7) B76(0-7) B77(0-7)

                    const __m256i rhs_hbit_0123_11 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_0123_3, m3b), 4); //B10(8-15) B11(8-15) B12(8-15) B13(8-15)
                    const __m256i rhs_hbit_0123_31 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_3, 2), m3b), 4); //B30(8-15) B31(8-15) B32(8-15) B33(8-15)
                    const __m256i rhs_hbit_0123_51 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_3, 4), m3b), 4); //B50(8-15) B51(8-15) B52(8-15) B53(8-15)
                    const __m256i rhs_hbit_0123_71 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_0123_3, 6), m3b), 4); //B70(8-15) B71(8-15) B72(8-15) B73(8-15)

                    const __m256i rhs_hbit_4567_11 = _mm256_slli_epi16(_mm256_and_si256(rhs_raw_hbit_4567_3, m3b), 4); //B14(8-15) B15(8-15) B16(8-15) B17(8-15)
                    const __m256i rhs_hbit_4567_31 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_3, 2), m3b), 4); //B34(8-15) B35(8-15) B36(8-15) B37(8-15)
                    const __m256i rhs_hbit_4567_51 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_3, 4), m3b), 4); //B54(8-15) B55(8-15) B56(8-15) B57(8-15)
                    const __m256i rhs_hbit_4567_71 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_hbit_4567_3, 6), m3b), 4); //B74(8-15) B75(8-15) B76(8-15) B77(8-15)

                    // Load the eight block_q6_k for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    // The lower and higher packed bits are loaded, unpacked and individual bytes representing 6 bits each are formed from the same
                    // They are blended/permuted for further mul mat operations within the pipeline
                    const __m256i rhs_raw_lbit_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + sb * 512));
                    const __m256i rhs_raw_lbit_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 32 + sb * 512));
                    const __m256i rhs_raw_lbit_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 64 + sb * 512));
                    const __m256i rhs_raw_lbit_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 96 + sb * 512));
                    const __m256i rhs_raw_lbit_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 128 + sb * 512));
                    const __m256i rhs_raw_lbit_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 160 + sb * 512));
                    const __m256i rhs_raw_lbit_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 192 + sb * 512));
                    const __m256i rhs_raw_lbit_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 224 + sb * 512));
                    const __m256i rhs_raw_lbit_0123_4 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 256 + sb * 512));
                    const __m256i rhs_raw_lbit_4567_4 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 288 + sb * 512));
                    const __m256i rhs_raw_lbit_0123_5 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 320 + sb * 512));
                    const __m256i rhs_raw_lbit_4567_5 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 352 + sb * 512));
                    const __m256i rhs_raw_lbit_0123_6 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 384 + sb * 512));
                    const __m256i rhs_raw_lbit_4567_6 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 416 + sb * 512));
                    const __m256i rhs_raw_lbit_0123_7 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 448 + sb * 512));
                    const __m256i rhs_raw_lbit_4567_7 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].ql + 480 + sb * 512));

                    // Comments indicate the indices of elements from individual super block in non interleaved fashion
                    // Index : 0 -7, 64 - 71
                    const __m256i rhs_vec_0123_00 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_0, m4b), rhs_hbit_0123_00);
                    const __m256i rhs_vec_0123_40 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_0, 4), m4b), rhs_hbit_0123_40);

                    const __m256i rhs_vec_4567_00 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_0, m4b), rhs_hbit_4567_00);
                    const __m256i rhs_vec_4567_40 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_0, 4), m4b), rhs_hbit_4567_40);

                    // Index : 8 - 15, 72 - 79
                    const __m256i rhs_vec_0123_01 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_1, m4b), rhs_hbit_0123_01);
                    const __m256i rhs_vec_0123_41 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_1, 4), m4b), rhs_hbit_0123_41);

                    const __m256i rhs_vec_4567_01 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_1, m4b), rhs_hbit_4567_01);
                    const __m256i rhs_vec_4567_41 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_1, 4), m4b), rhs_hbit_4567_41);

                    // Index : 16 - 23, 80 - 87
                    const __m256i rhs_vec_0123_10 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_2, m4b), rhs_hbit_0123_10);
                    const __m256i rhs_vec_0123_50 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_2, 4), m4b), rhs_hbit_0123_50);

                    const __m256i rhs_vec_4567_10 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_2, m4b), rhs_hbit_4567_10);
                    const __m256i rhs_vec_4567_50 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_2, 4), m4b), rhs_hbit_4567_50);

                    // Index : 24 - 31, 88 - 95
                    const __m256i rhs_vec_0123_11 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_3, m4b), rhs_hbit_0123_11);
                    const __m256i rhs_vec_0123_51 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_3, 4), m4b), rhs_hbit_0123_51);

                    const __m256i rhs_vec_4567_11 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_3, m4b), rhs_hbit_4567_11);
                    const __m256i rhs_vec_4567_51 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_3, 4), m4b), rhs_hbit_4567_51);

                    // Index : 32 - 39, 96 - 103
                    const __m256i rhs_vec_0123_20 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_4, m4b), rhs_hbit_0123_20);
                    const __m256i rhs_vec_0123_60 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_4, 4), m4b), rhs_hbit_0123_60);

                    const __m256i rhs_vec_4567_20 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_4, m4b), rhs_hbit_4567_20);
                    const __m256i rhs_vec_4567_60 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_4, 4), m4b), rhs_hbit_4567_60);

                    // Index : 40 - 47, 104 - 111
                    const __m256i rhs_vec_0123_21 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_5, m4b), rhs_hbit_0123_21);
                    const __m256i rhs_vec_0123_61 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_5, 4), m4b), rhs_hbit_0123_61);

                    const __m256i rhs_vec_4567_21 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_5, m4b), rhs_hbit_4567_21);
                    const __m256i rhs_vec_4567_61 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_5, 4), m4b), rhs_hbit_4567_61);

                    // Index : 48 - 55, 112 - 119
                    const __m256i rhs_vec_0123_30 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_6, m4b), rhs_hbit_0123_30);
                    const __m256i rhs_vec_0123_70 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_6, 4), m4b), rhs_hbit_0123_70);

                    const __m256i rhs_vec_4567_30 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_6, m4b), rhs_hbit_4567_30);
                    const __m256i rhs_vec_4567_70 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_6, 4), m4b), rhs_hbit_4567_70);

                    // Index : 56 - 63, 120 - 127
                    const __m256i rhs_vec_0123_31 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_0123_7, m4b), rhs_hbit_0123_31);
                    const __m256i rhs_vec_0123_71 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_0123_7, 4), m4b), rhs_hbit_0123_71);

                    const __m256i rhs_vec_4567_31 = _mm256_or_si256(_mm256_and_si256(rhs_raw_lbit_4567_7, m4b), rhs_hbit_4567_31);
                    const __m256i rhs_vec_4567_71 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(rhs_raw_lbit_4567_7, 4), m4b), rhs_hbit_4567_71);

                    //Scales of corresponding sub blocks from different Q6_K structures are stored together
                    //s00 s01 s10 s11 s20 s21 s30 s31 s40 s41 s50 s51 s60 s61 s70 s71
                    const __m128i scales_01 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + sb * 64));
                    const __m128i scales_23 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 16 + sb * 64));
                    const __m128i scales_45 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 32 + sb * 64));
                    const __m128i scales_67 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 48 + sb * 64));

                    // Scales of sub blocks in the sb loop
                    // Scales of the 0th sub block from each super block
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(scales_01, scalemask1);
                    __m256i scales_0 = _mm256_cvtepi8_epi16(scales_rearrange_0);

                    // Scales of the 1st sub block from each super block
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(scales_01, scalemask2);
                    __m256i scales_1 = _mm256_cvtepi8_epi16(scales_rearrange_1);

                    // Scales of the 2nd sub block from each super block
                    __m128i scales_rearrange_2 = _mm_shuffle_epi8(scales_23, scalemask1);
                    __m256i scales_2 = _mm256_cvtepi8_epi16(scales_rearrange_2);

                    // Scales of the 3rd sub block from each super block
                    __m128i scales_rearrange_3 = _mm_shuffle_epi8(scales_23, scalemask2);
                    __m256i scales_3 = _mm256_cvtepi8_epi16(scales_rearrange_3);

                    // Scales of the 4th sub block from each super block
                    __m128i scales_rearrange_4 = _mm_shuffle_epi8(scales_45, scalemask1);
                    __m256i scales_4 = _mm256_cvtepi8_epi16(scales_rearrange_4);

                    // Scales of the 5th sub block from each super block
                    __m128i scales_rearrange_5 = _mm_shuffle_epi8(scales_45, scalemask2);
                    __m256i scales_5 = _mm256_cvtepi8_epi16(scales_rearrange_5);

                    // Scales of the 6th sub block from each super block
                    __m128i scales_rearrange_6 = _mm_shuffle_epi8(scales_67, scalemask1);
                    __m256i scales_6 = _mm256_cvtepi8_epi16(scales_rearrange_6);

                    // Scales of the 7th sub block from each super block
                    __m128i scales_rearrange_7 = _mm_shuffle_epi8(scales_67, scalemask2);
                    __m256i scales_7 = _mm256_cvtepi8_epi16(scales_rearrange_7);

                    // Load the sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    __m256i lhs_vec_0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 128)));
                    __m256i lhs_vec_1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 128)));
                    __m256i lhs_vec_2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 128)));
                    __m256i lhs_vec_3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 128)));
                    __m256i lhs_vec_4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 64 + sb * 128)));
                    __m256i lhs_vec_5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 80 + sb * 128)));
                    __m256i lhs_vec_6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 96 + sb * 128)));
                    __m256i lhs_vec_7 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 112 + sb * 128)));

                    lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0, 0);
                    lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1, 0);
                    lhs_vec_2 = _mm256_permute2f128_si256(lhs_vec_2, lhs_vec_2, 0);
                    lhs_vec_3 = _mm256_permute2f128_si256(lhs_vec_3, lhs_vec_3, 0);
                    lhs_vec_4 = _mm256_permute2f128_si256(lhs_vec_4, lhs_vec_4, 0);
                    lhs_vec_5 = _mm256_permute2f128_si256(lhs_vec_5, lhs_vec_5, 0);
                    lhs_vec_6 = _mm256_permute2f128_si256(lhs_vec_6, lhs_vec_6, 0);
                    lhs_vec_7 = _mm256_permute2f128_si256(lhs_vec_7, lhs_vec_7, 0);

                    // Multiply Q8 quants with bytes valued 32 - Subtracted later as an adjustment for 6 bit quantization
                    __m256i lhs_vec_s_0 = _mm256_maddubs_epi16(m32s, lhs_vec_0);
                    __m256i lhs_vec_s_1 = _mm256_maddubs_epi16(m32s, lhs_vec_1);
                    __m256i lhs_vec_s_2 = _mm256_maddubs_epi16(m32s, lhs_vec_2);
                    __m256i lhs_vec_s_3 = _mm256_maddubs_epi16(m32s, lhs_vec_3);
                    __m256i lhs_vec_s_4 = _mm256_maddubs_epi16(m32s, lhs_vec_4);
                    __m256i lhs_vec_s_5 = _mm256_maddubs_epi16(m32s, lhs_vec_5);
                    __m256i lhs_vec_s_6 = _mm256_maddubs_epi16(m32s, lhs_vec_6);
                    __m256i lhs_vec_s_7 = _mm256_maddubs_epi16(m32s, lhs_vec_7);

                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();
                    __m256i iacc_2 = _mm256_setzero_si256();
                    __m256i iacc_3 = _mm256_setzero_si256();
                    __m256i iacc_4 = _mm256_setzero_si256();
                    __m256i iacc_5 = _mm256_setzero_si256();
                    __m256i iacc_6 = _mm256_setzero_si256();
                    __m256i iacc_7 = _mm256_setzero_si256();

                    // Dot product done within 32 bit lanes and accumulated in the same vector
                    // First done for 0th sub block and then for seven (1st - 7th) other sub blocks processed for each sb (sb < QK_K/128 loop)
                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                    // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                    // B0(8-11) B4(8-11) B1(8-11) B5(8-11) B2(8-11) B6(8-11) B3(8-11) B7(8-11) with A0(8-11)
                    // B0(12-15) B4(12-15) B1(12-15) B5(12-15) B2(12-15) B6(12-15) B3(12-15) B7(12-15) with A0(12-15)

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 0)), _mm256_shuffle_epi32(lhs_vec_s_0, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_0, 85)), _mm256_shuffle_epi32(lhs_vec_s_0, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 170)), _mm256_shuffle_epi32(lhs_vec_s_0, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_0, 255)), _mm256_shuffle_epi32(lhs_vec_s_0, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 0)), _mm256_shuffle_epi32(lhs_vec_s_1, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_1, 85)), _mm256_shuffle_epi32(lhs_vec_s_1, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 170)), _mm256_shuffle_epi32(lhs_vec_s_1, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_1, 255)), _mm256_shuffle_epi32(lhs_vec_s_1, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_20 ,_mm256_shuffle_epi32(rhs_vec_4567_20, 177), 170), _mm256_shuffle_epi32(lhs_vec_2, 0)), _mm256_shuffle_epi32(lhs_vec_s_2, 0)));
                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_20, 177) ,rhs_vec_4567_20, 170), _mm256_shuffle_epi32(lhs_vec_2, 85)), _mm256_shuffle_epi32(lhs_vec_s_2, 85)));

                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_21 ,_mm256_shuffle_epi32(rhs_vec_4567_21, 177), 170), _mm256_shuffle_epi32(lhs_vec_2, 170)), _mm256_shuffle_epi32(lhs_vec_s_2, 170)));
                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_21, 177) ,rhs_vec_4567_21, 170), _mm256_shuffle_epi32(lhs_vec_2, 255)), _mm256_shuffle_epi32(lhs_vec_s_2, 255)));

                    iacc_2 = _mm256_madd_epi16(iacc_2, scales_2);

                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_30 ,_mm256_shuffle_epi32(rhs_vec_4567_30, 177), 170), _mm256_shuffle_epi32(lhs_vec_3, 0)), _mm256_shuffle_epi32(lhs_vec_s_3, 0)));
                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_30, 177) ,rhs_vec_4567_30, 170), _mm256_shuffle_epi32(lhs_vec_3, 85)), _mm256_shuffle_epi32(lhs_vec_s_3, 85)));

                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_31 ,_mm256_shuffle_epi32(rhs_vec_4567_31, 177), 170), _mm256_shuffle_epi32(lhs_vec_3, 170)), _mm256_shuffle_epi32(lhs_vec_s_3, 170)));
                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_31, 177) ,rhs_vec_4567_31, 170), _mm256_shuffle_epi32(lhs_vec_3, 255)), _mm256_shuffle_epi32(lhs_vec_s_3, 255)));

                    iacc_3 = _mm256_madd_epi16(iacc_3, scales_3);

                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_40 ,_mm256_shuffle_epi32(rhs_vec_4567_40, 177), 170), _mm256_shuffle_epi32(lhs_vec_4, 0)), _mm256_shuffle_epi32(lhs_vec_s_4, 0)));
                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_40, 177) ,rhs_vec_4567_40, 170), _mm256_shuffle_epi32(lhs_vec_4, 85)), _mm256_shuffle_epi32(lhs_vec_s_4, 85)));

                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_41 ,_mm256_shuffle_epi32(rhs_vec_4567_41, 177), 170), _mm256_shuffle_epi32(lhs_vec_4, 170)), _mm256_shuffle_epi32(lhs_vec_s_4, 170)));
                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_41, 177) ,rhs_vec_4567_41, 170), _mm256_shuffle_epi32(lhs_vec_4, 255)), _mm256_shuffle_epi32(lhs_vec_s_4, 255)));

                    iacc_4 = _mm256_madd_epi16(iacc_4, scales_4);

                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_50 ,_mm256_shuffle_epi32(rhs_vec_4567_50, 177), 170), _mm256_shuffle_epi32(lhs_vec_5, 0)), _mm256_shuffle_epi32(lhs_vec_s_5, 0)));
                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_50, 177) ,rhs_vec_4567_50, 170), _mm256_shuffle_epi32(lhs_vec_5, 85)), _mm256_shuffle_epi32(lhs_vec_s_5, 85)));

                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_51 ,_mm256_shuffle_epi32(rhs_vec_4567_51, 177), 170), _mm256_shuffle_epi32(lhs_vec_5, 170)), _mm256_shuffle_epi32(lhs_vec_s_5, 170)));
                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_51, 177) ,rhs_vec_4567_51, 170), _mm256_shuffle_epi32(lhs_vec_5, 255)), _mm256_shuffle_epi32(lhs_vec_s_5, 255)));

                    iacc_5 = _mm256_madd_epi16(iacc_5, scales_5);

                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_60 ,_mm256_shuffle_epi32(rhs_vec_4567_60, 177), 170), _mm256_shuffle_epi32(lhs_vec_6, 0)), _mm256_shuffle_epi32(lhs_vec_s_6, 0)));
                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_60, 177) ,rhs_vec_4567_60, 170), _mm256_shuffle_epi32(lhs_vec_6, 85)), _mm256_shuffle_epi32(lhs_vec_s_6, 85)));

                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_61 ,_mm256_shuffle_epi32(rhs_vec_4567_61, 177), 170), _mm256_shuffle_epi32(lhs_vec_6, 170)), _mm256_shuffle_epi32(lhs_vec_s_6, 170)));
                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_61, 177) ,rhs_vec_4567_61, 170), _mm256_shuffle_epi32(lhs_vec_6, 255)), _mm256_shuffle_epi32(lhs_vec_s_6, 255)));

                    iacc_6 = _mm256_madd_epi16(iacc_6, scales_6);

                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_70 ,_mm256_shuffle_epi32(rhs_vec_4567_70, 177), 170), _mm256_shuffle_epi32(lhs_vec_7, 0)), _mm256_shuffle_epi32(lhs_vec_s_7, 0)));
                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_70, 177) ,rhs_vec_4567_70, 170), _mm256_shuffle_epi32(lhs_vec_7, 85)), _mm256_shuffle_epi32(lhs_vec_s_7, 85)));

                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_71 ,_mm256_shuffle_epi32(rhs_vec_4567_71, 177), 170), _mm256_shuffle_epi32(lhs_vec_7, 170)), _mm256_shuffle_epi32(lhs_vec_s_7, 170)));
                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_71, 177) ,rhs_vec_4567_71, 170), _mm256_shuffle_epi32(lhs_vec_7, 255)), _mm256_shuffle_epi32(lhs_vec_s_7, 255)));

                    iacc_7 = _mm256_madd_epi16(iacc_7, scales_7);

                    // Accumulate the iacc value for one sb
                    __m256i iacc_sb = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_0, iacc_1), _mm256_add_epi32(iacc_2, iacc_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_4, iacc_5), _mm256_add_epi32(iacc_6, iacc_7)));

                    // Accumulate for the complete block
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                }

                //Multiply-Add with scale values for complete super block
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            }
            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);

        }
    }
#else

    ggml_gemv_q6_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);

#endif
}
int main(){block_q6_Kx8 w{};for(auto&d:w.d)d=0x3c00;std::memset(w.scales,127,sizeof(w.scales));std::memset(w.ql,255,sizeof(w.ql));std::memset(w.qh,255,sizeof(w.qh));
 for(int act:{0,1,16,64,127,-128}){block_q8_K x{};x.d=1.f;for(auto&q:x.qs)q=act;for(auto&q:x.bsums)q=act*16;float out[8]{};probe_q6_gemv(256,out,0,&w,&x,1,8);float expected=float(256*31*127*act);int bad=0;for(float v:out)bad+=(v!=expected);std::printf("{\"activation\":%d,\"expected\":%.9g,\"actual0\":%.9g,\"wrong_columns\":%d}\n",act,expected,out[0],bad);}
}
