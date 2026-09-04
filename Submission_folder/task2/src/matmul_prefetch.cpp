#include <immintrin.h>

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 256;
    const int PREFETCH_DIST = 64;

    for (int kk = 0; kk < K; kk += BK) {
        int k_end = (kk + BK < K) ? (kk + BK) : K;

        for (int ii = 0; ii < M; ii += BM) {
            int i_end = (ii + BM < M) ? (ii + BM) : M;

            for (int jj = 0; jj < N; jj += BN) {
                int j_end = (jj + BN < N) ? (jj + BN) : N;

                for (int i = ii; i < i_end; ++i) {
                    const float* a = A + static_cast<long>(i) * lda;

                    for (int j = jj; j < j_end; ++j) {
                        const float* b = B + static_cast<long>(j) * ldb;

                        __m256 vacc = (kk == 0) ? _mm256_setzero_ps() 
                                               : _mm256_set1_ps(0.0f); 
                        float acc = 0.0f;

                        int p = kk;
                        for (; p + 7 < k_end; p += 8) {
                            _mm_prefetch((const char*)(a + p + PREFETCH_DIST), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b + p + PREFETCH_DIST), _MM_HINT_T0);

                            __m256 va = _mm256_loadu_ps(a + p);
                            __m256 vb = _mm256_loadu_ps(b + p);
                            vacc = _mm256_fmadd_ps(va, vb, vacc);
                        }

                        __m128 low = _mm256_castps256_ps128(vacc);
                        __m128 high = _mm256_extractf128_ps(vacc, 1);
                        low = _mm_add_ps(low, high);
                        low = _mm_hadd_ps(low, low);
                        low = _mm_hadd_ps(low, low);
                        acc = _mm_cvtss_f32(low);

                        for (; p < k_end; ++p) {
                            acc += a[p] * b[p];
                        }
                        if (kk == 0) {
                            C[static_cast<long>(i) * ldc + j] = acc;
                        } else {
                            C[static_cast<long>(i) * ldc + j] += acc;
                        }
                    }
                }
            }
        }
    }
}