// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            out[oy*W + ox] = 0.0f;
        }
    }
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            __m256 vk = _mm256_set1_ps(ker[ky * K + kx]);
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy+ky)*in_stride + kx;
                float* out_row = out + oy*W;
                int ox = 0;
                for (; ox+7 < W; ox += 8) {
                    __m256 vin = _mm256_loadu_ps(in_row + ox);
                    __m256 vout = _mm256_loadu_ps(out_row + ox);
                    vout = _mm256_add_ps(vout, _mm256_mul_ps(vin, vk));
                    _mm256_storeu_ps(out_row + ox, vout);
                }
                for (; ox < W; ++ox) {
                    out_row[ox] += in_row[ox] * ker[ky*K + kx];
                }
            }
        }
    }
}
