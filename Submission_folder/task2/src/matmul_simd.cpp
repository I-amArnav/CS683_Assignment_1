// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics

#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    constexpr int TILE_N = 4;
    constexpr int SIMD_K = 8;

    for (int i = 0; i < M; ++i) {
        const float* a_row = A + static_cast<long>(i) * lda;
        float* c_row = C + static_cast<long>(i) * ldc;
        int j = 0;

        for (; j + TILE_N <= N; j += TILE_N) {

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;
            int p = 0;

            for (; p + SIMD_K <= K; p += SIMD_K) {
                const __m256 a = _mm256_loadu_ps(a_row + p);
                const __m256 bv0 = _mm256_loadu_ps(b0 + p);
                const __m256 bv1 = _mm256_loadu_ps(b1 + p);
                const __m256 bv2 = _mm256_loadu_ps(b2 + p);
                const __m256 bv3 = _mm256_loadu_ps(b3 + p);
                acc0 = _mm256_fmadd_ps(a, bv0, acc0);
                acc1 = _mm256_fmadd_ps(a, bv1, acc1);
                acc2 = _mm256_fmadd_ps(a, bv2, acc2);
                acc3 = _mm256_fmadd_ps(a, bv3, acc3);
            }

            float tmp0[8];
            float tmp1[8];
            float tmp2[8];
            float tmp3[8];
            _mm256_storeu_ps(tmp0, acc0);
            _mm256_storeu_ps(tmp1, acc1);
            _mm256_storeu_ps(tmp2, acc2);
            _mm256_storeu_ps(tmp3, acc3);

            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;

            for (int x = 0; x < 8; ++x) {
                sum0 += tmp0[x];
                sum1 += tmp1[x];
                sum2 += tmp2[x];
                sum3 += tmp3[x];
            }

            for (; p < K; ++p) {
                const float a = a_row[p];

                sum0 += a * b0[p];
                sum1 += a * b1[p];
                sum2 += a * b2[p];
                sum3 += a * b3[p];
            }

            c_row[j + 0] = sum0;
            c_row[j + 1] = sum1;
            c_row[j + 2] = sum2;
            c_row[j + 3] = sum3;
        }

        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + SIMD_K <= K; p += SIMD_K) {
                const __m256 a = _mm256_loadu_ps(a_row + p);
                const __m256 bv = _mm256_loadu_ps(b + p);
                acc = _mm256_fmadd_ps(a, bv, acc);
            }

            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            float sum = 0.0f;
            for (int x = 0; x < 8; ++x) {
                sum += tmp[x];
            }
            for (; p < K; ++p) {
                sum += a_row[p] * b[p];
            }
            c_row[j] = sum;
        }
    }
}