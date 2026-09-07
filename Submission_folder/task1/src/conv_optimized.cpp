// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int tile_h = 16;
    const int tile_w = 256;
    const int unroll_w_factor = 32;

    for (int oy0=0; oy0 < H; oy0 += tile_h) {
        const int oy_end = (oy0 + tile_h < H) ? oy0 + tile_h : H;

        for (int ox0=0; ox0 < W; ox0 += tile_w) {
            const int ox_end = (ox0 + tile_w < W) ? ox0 + tile_w : W;

            for (int oy = oy0; oy < oy_end; ++oy) {
                int ox = ox0;
                for (; ox + unroll_w_factor <= ox_end ; ox += unroll_w_factor) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();


                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row = in + (oy + ky)*in_stride;
                        for (int kx = 0; kx < K; ++kx) {
                            const __m256 w = _mm256_set1_ps(ker[ky*K + kx]);
                            const float* src = in_row + ox + kx;
                            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(src), w, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 8), w, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 16), w, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 24), w, acc3);
                        }
                    }

                    float* dst = out + oy*W + ox;
                    _mm256_storeu_ps(dst, acc0);
                    _mm256_storeu_ps(dst + 8, acc1);
                    _mm256_storeu_ps(dst + 16, acc2);
                    _mm256_storeu_ps(dst + 24, acc3);
                }
                for (; ox < ox_end; ox += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row = in + (oy + ky)*in_stride;
                        for (int kx = 0; kx < K; ++kx) {
                            const __m256 w = _mm256_set1_ps(ker[ky*K + kx]);
                            const __m256 pixels = _mm256_loadu_ps(in_row + ox + kx);
                            acc = _mm256_fmadd_ps(pixels, w, acc);
                        }
                    }
                    _mm256_storeu_ps(out + oy*W + ox, acc);
                }
            }
        }
    }
}
