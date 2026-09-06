#include <immintrin.h>

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    const int BM = 32;
    const int BN = 32;
    const int BK = 32;
    
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

                        _mm_prefetch((const char*)(a + kk + PREFETCH_DIST), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b + kk + PREFETCH_DIST), _MM_HINT_T0);

                        float acc = 0.0f;

                        for (int p = kk; p < k_end; ++p) {
                            acc += a[p] * b[p];
                        }

                        long c_idx = static_cast<long>(i) * ldc + j;
                        if (kk == 0) {
                            C[c_idx] = acc;
                        } else {
                            C[c_idx] += acc;
                        }
                    }
                }
            }
        }
    }
}