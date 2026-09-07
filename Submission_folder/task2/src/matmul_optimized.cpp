#include <immintrin.h>
#include <cstdint>
#include <algorithm>

static inline void helper_3x4(const float* __restrict__ a0, const float* __restrict__ a1, const float* __restrict__ a2,
    const float* __restrict__ b0, const float* __restrict__ b1, const float* __restrict__ b2, const float* __restrict__ b3,
    float* __restrict__ c0, float* __restrict__ c1, float* __restrict__ c2,
    int K, bool beta_zero) {

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps(), c03 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps(), c23 = _mm256_setzero_ps();

    int p = 0;    
    for (; p + 15 < K; p += 16) {
        _mm_prefetch(reinterpret_cast<const char*>(a0 + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a1 + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a2 + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b0 + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b1 + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b2 + 32), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b3 + 32), _MM_HINT_T0);

        __m256 va0 = _mm256_loadu_ps(a0);
        __m256 va1 = _mm256_loadu_ps(a1);
        __m256 va2 = _mm256_loadu_ps(a2);
        __m256 vb;
        
        vb = _mm256_loadu_ps(b0); 
        c00 = _mm256_fmadd_ps(va0, vb, c00);
        c10 = _mm256_fmadd_ps(va1, vb, c10);
        c20 = _mm256_fmadd_ps(va2, vb, c20);

        vb = _mm256_loadu_ps(b1); 
        c01 = _mm256_fmadd_ps(va0, vb, c01);
        c11 = _mm256_fmadd_ps(va1, vb, c11);
        c21 = _mm256_fmadd_ps(va2, vb, c21);

        vb = _mm256_loadu_ps(b2); 
        c02 = _mm256_fmadd_ps(va0, vb, c02);
        c12 = _mm256_fmadd_ps(va1, vb, c12);
        c22 = _mm256_fmadd_ps(va2, vb, c22);

        vb = _mm256_loadu_ps(b3); 
        c03 = _mm256_fmadd_ps(va0, vb, c03);
        c13 = _mm256_fmadd_ps(va1, vb, c13);
        c23 = _mm256_fmadd_ps(va2, vb, c23);

        va0 = _mm256_loadu_ps(a0 + 8);
        va1 = _mm256_loadu_ps(a1 + 8);
        va2 = _mm256_loadu_ps(a2 + 8);
        
        vb = _mm256_loadu_ps(b0 + 8); 
        c00 = _mm256_fmadd_ps(va0, vb, c00);
        c10 = _mm256_fmadd_ps(va1, vb, c10);
        c20 = _mm256_fmadd_ps(va2, vb, c20);

        vb = _mm256_loadu_ps(b1 + 8); 
        c01 = _mm256_fmadd_ps(va0, vb, c01);
        c11 = _mm256_fmadd_ps(va1, vb, c11);
        c21 = _mm256_fmadd_ps(va2, vb, c21);

        vb = _mm256_loadu_ps(b2 + 8); 
        c02 = _mm256_fmadd_ps(va0, vb, c02);
        c12 = _mm256_fmadd_ps(va1, vb, c12);
        c22 = _mm256_fmadd_ps(va2, vb, c22);

        vb = _mm256_loadu_ps(b3 + 8); 
        c03 = _mm256_fmadd_ps(va0, vb, c03);
        c13 = _mm256_fmadd_ps(va1, vb, c13);
        c23 = _mm256_fmadd_ps(va2, vb, c23);

        a0 += 16;a1 += 16;a2 += 16;b0 += 16; b1+= 16; b2 += 16;b3+= 16;
    }

    if (p+7 < K) {
        __m256 va0 = _mm256_loadu_ps(a0);
        __m256 va1 = _mm256_loadu_ps(a1);
        __m256 va2 = _mm256_loadu_ps(a2);
        __m256 vb;
        
        vb = _mm256_loadu_ps(b0);
        c00 = _mm256_fmadd_ps(va0, vb, c00);
        c10 = _mm256_fmadd_ps(va1, vb, c10);
        c20 = _mm256_fmadd_ps(va2, vb, c20);

        vb = _mm256_loadu_ps(b1);
        c01 = _mm256_fmadd_ps(va0, vb, c01);
        c11 = _mm256_fmadd_ps(va1, vb, c11);
        c21 = _mm256_fmadd_ps(va2, vb, c21);

        vb = _mm256_loadu_ps(b2);
        c02 = _mm256_fmadd_ps(va0, vb, c02);
        c12 = _mm256_fmadd_ps(va1, vb, c12);
        c22 = _mm256_fmadd_ps(va2, vb, c22);

        vb = _mm256_loadu_ps(b3);
        c03 = _mm256_fmadd_ps(va0, vb, c03);
        c13 = _mm256_fmadd_ps(va1, vb, c13);
        c23 = _mm256_fmadd_ps(va2, vb, c23);

        p += 8;a0 += 8;a1+= 8;a2+= 8;
        b0 += 8; b1+= 8; b2+= 8; b3+= 8;
    }

    auto hsum = [](__m256 v) -> float {
        __m128 vlow = _mm256_castps256_ps128(v);
        __m128 vhigh = _mm256_extractf128_ps(v, 1);
        __m128 v128 = _mm_add_ps(vlow, vhigh);
        __m128 shuf = _mm_movehdup_ps(v128);
        __m128 sums = _mm_add_ps(v128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    };

    float s00 = hsum(c00), s01 = hsum(c01), s02 = hsum(c02), s03 = hsum(c03);
    float s10 = hsum(c10), s11 = hsum(c11), s12 = hsum(c12), s13 = hsum(c13);
    float s20 = hsum(c20), s21 = hsum(c21), s22 = hsum(c22), s23 = hsum(c23);

    for (; p < K; ++p) {
        float va0 = *a0++, va1 = *a1++, va2 = *a2++;
        float vb0 = *b0++, vb1 = *b1++, vb2 = *b2++, vb3 = *b3++;
        s00 += va0 * vb0; s01 += va0 * vb1; s02 += va0 * vb2; s03 += va0 * vb3;
        s10 += va1 * vb0; s11 += va1 * vb1; s12 += va1 * vb2; s13 += va1 * vb3;
        s20 += va2 * vb0; s21 += va2 * vb1; s22 += va2 * vb2; s23 += va2 * vb3;
    }

    if (beta_zero) {
        c0[0] = s00; c0[1] = s01; c0[2] = s02; c0[3] = s03;
        c1[0] = s10; c1[1] = s11; c1[2] = s12; c1[3] = s13;
        c2[0] = s20; c2[1] = s21; c2[2] = s22; c2[3] = s23;
    } else {
        c0[0] += s00; c0[1] += s01; c0[2] += s02; c0[3] += s03;
        c1[0] += s10; c1[1] += s11; c1[2] += s12; c1[3] += s13;
        c2[0] += s20; c2[1] += s21; c2[2] += s22; c2[3] += s23;
    }
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc){

    constexpr int MC = 120;
    constexpr int NC = 128;
    constexpr int KC = 256;

    for(int jc = 0; jc < N; jc += NC) {
        for(int pc = 0; pc < K; pc += KC) {
            for(int ic = 0; ic < M; ic += MC) {
                int mc = std::min(M - ic, MC);
                int nc = std::min(N - jc, NC);
                int kc = std::min(K - pc, KC);
                bool beta_zero = (pc == 0);

                int i=0;
                for(; i+2 < mc; i+=3) {
                    int j = 0;
                    for(; j+3 < nc; j+=4) {
                        helper_3x4(A + static_cast<int64_t>(ic + i + 0) * lda + pc,
                                    A + static_cast<int64_t>(ic + i + 1) * lda + pc,
                                    A + static_cast<int64_t>(ic + i + 2) * lda + pc,
                                    B + static_cast<int64_t>(jc + j + 0) * ldb + pc,
                                    B + static_cast<int64_t>(jc + j + 1) * ldb + pc,
                                    B + static_cast<int64_t>(jc + j + 2) * ldb + pc,
                                    B + static_cast<int64_t>(jc + j + 3) * ldb + pc,
                                    C + static_cast<int64_t>(ic + i + 0) * ldc + (jc+j),
                                    C + static_cast<int64_t>(ic + i + 1) * ldc + (jc+j),
                                    C + static_cast<int64_t>(ic + i + 2) * ldc + (jc+j),kc, beta_zero);
                    }
                    for(; j < nc; ++j) {
                        for(int r = 0; r < 3; ++r) {
                            float sum = 0.0f;
                            const float* a_ptr = A + static_cast<int64_t>(ic + i + r)*lda + pc;
                            const float* b_ptr = B + static_cast<int64_t>(jc + j)*ldb + pc;
                            for (int p=0; p < kc; ++p) sum += a_ptr[p]*b_ptr[p];

                            float* cp = C + static_cast<int64_t>(ic + i + r) * ldc + (jc + j);
                            if (beta_zero) 
                                *cp = sum; 
                            else *cp += sum;
                        }
                    }
                }
                for(; i < mc; ++i) {
                    for(int j=0; j < nc; ++j) {
                        float sum = 0.0f;
                        const float* a_ptr = A + static_cast<int64_t>(ic + i)*lda + pc;
                        const float* b_ptr = B + static_cast<int64_t>(jc + j)*ldb + pc;

                        for (int p = 0; p < kc; ++p) sum += a_ptr[p]*b_ptr[p];                        
                        float* cp = C + static_cast<int64_t>(ic + i)*ldc + (jc+j);
                        if (beta_zero)
                            *cp = sum;
                        else *cp += sum;
                    }
                }
            }
        }
    }
}
