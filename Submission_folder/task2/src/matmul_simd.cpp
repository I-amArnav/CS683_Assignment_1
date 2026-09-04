#include <immintrin.h>

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    
    int i = 0;
    for (; i+3 < M; i+=4) {
        int j = 0;
        for (; j+1 < N; j+=2) {
            
            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            const float* a3 = A + static_cast<long>(i + 3) * lda;

            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;

            __m256 acc00 = _mm256_setzero_ps();
            __m256 acc01 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps();
            __m256 acc11 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps();
            __m256 acc21 = _mm256_setzero_ps();
            __m256 acc30 = _mm256_setzero_ps();
            __m256 acc31 = _mm256_setzero_ps();

            int p = 0;
            for (; p+7 < K; p+=8) {
                __m256 va0 = _mm256_loadu_ps(a0 + p);
                __m256 va1 = _mm256_loadu_ps(a1 + p);
                __m256 va2 = _mm256_loadu_ps(a2 + p);
                __m256 va3 = _mm256_loadu_ps(a3 + p);

                __m256 vb0 = _mm256_loadu_ps(b0 + p);
                __m256 vb1 = _mm256_loadu_ps(b1 + p);

                acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
            }

            auto hsum = [](__m256 v) -> float {
                __m128 low = _mm256_castps256_ps128(v);
                __m128 high = _mm256_extractf128_ps(v, 1);
                low = _mm_add_ps(low, high);
                low = _mm_hadd_ps(low, low);
                low = _mm_hadd_ps(low, low);
                return _mm_cvtss_f32(low);
            };

            float s00 = hsum(acc00), s01 = hsum(acc01);
            float s10 = hsum(acc10), s11 = hsum(acc11);
            float s20 = hsum(acc20), s21 = hsum(acc21);
            float s30 = hsum(acc30), s31 = hsum(acc31);

            for (; p < K; ++p) {
                s00 += a0[p] * b0[p]; s01 += a0[p] * b1[p];
                s10 += a1[p] * b0[p]; s11 += a1[p] * b1[p];
                s20 += a2[p] * b0[p]; s21 += a2[p] * b1[p];
                s30 += a3[p] * b0[p]; s31 += a3[p] * b1[p];
            }

            C[static_cast<long>(i + 0) * ldc + (j + 0)] = s00;
            C[static_cast<long>(i + 0) * ldc + (j + 1)] = s01;
            C[static_cast<long>(i + 1) * ldc + (j + 0)] = s10;
            C[static_cast<long>(i + 1) * ldc + (j + 1)] = s11;
            C[static_cast<long>(i + 2) * ldc + (j + 0)] = s20;
            C[static_cast<long>(i + 2) * ldc + (j + 1)] = s21;
            C[static_cast<long>(i + 3) * ldc + (j + 0)] = s30;
            C[static_cast<long>(i + 3) * ldc + (j + 1)] = s31;
        }

        for (; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            for (int r = 0; r < 4; ++r) {
                const float* a = A + static_cast<long>(i + r) * lda;
                float acc = 0.0f;
                for (int p = 0; p < K; ++p) acc += a[p] * b[p];
                C[static_cast<long>(i + r) * ldc + j] = acc;
            }
        }
    }

    for (; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) acc += a[p] * b[p];
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}