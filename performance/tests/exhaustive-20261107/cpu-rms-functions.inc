template <ggml_rms_norm_fuse_op FUSE_OP>
static void rms_before(
        const ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    const ggml_tensor * src0 = dst_rms_norm->src[0];
    const ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst_rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                const float mean  = sum/ne00;
                const float scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                assert(scale > 0.0f);

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    const int64_t i11 = i01 % ne11;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i13 = i03 % ne13;
                    const float * w = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } else {
                    memcpy(y, x, ne00 * sizeof(float));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}

template <ggml_rms_norm_fuse_op FUSE_OP>
static void rms_after(
        const ggml_compute_params * params,
        ggml_tensor * dst_rms_norm,
        ggml_tensor * dst_fused = nullptr) {

    const ggml_tensor * src0 = dst_rms_norm->src[0];
    const ggml_tensor * src1 = nullptr;
    ggml_tensor       * dst  = dst_rms_norm;

    if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
        src1 = (dst_fused->src[0] == dst_rms_norm) ? dst_fused->src[1] : dst_fused->src[0];
        dst  = dst_fused;
    }

    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    GGML_ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    GGML_TENSOR_BINARY_OP_LOCALS

    float eps;
    memcpy(&eps, dst_rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    // TODO: optimize
    for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            // Spread short leading rows over heads/batches, preserving every
            // row's double-precision sum and the ordinary full-worker path.
            const int64_t row_base = (i03*ne02 + i02)*ne01;
            const int64_t first_row = ne01 < nth ? (ith + nth - row_base % nth) % nth : ith;
            for (int64_t i01 = first_row; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                const float mean  = sum/ne00;
                const float scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                assert(scale > 0.0f);

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    const int64_t i11 = i01 % ne11;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i13 = i03 % ne13;
                    const float * w = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } else {
                    memcpy(y, x, ne00 * sizeof(float));
                    ggml_vec_scale_f32(ne00, y, scale);
                }
            }
        }
    }
}
