// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float kval = ker[ky * K + kx];
            if(kx==0 && ky==0){
                for (int oy = 0; oy < H; ++oy) {
                    const float* in_row = in + (oy + ky) * in_stride + kx;
                    float* out_row = out + oy * W;
                    int ox = 0;
                    for (; ox < W; ox += 8) {
                        out_row[ox + 0] = in_row[ox + 0] * kval;
                        out_row[ox + 1] = in_row[ox + 1] * kval;
                        out_row[ox + 2] = in_row[ox + 2] * kval;
                        out_row[ox + 3] = in_row[ox + 3] * kval;
                        out_row[ox + 4] = in_row[ox + 4] * kval;
                        out_row[ox + 5] = in_row[ox + 5] * kval;
                        out_row[ox + 6] = in_row[ox + 6] * kval;
                        out_row[ox + 7] = in_row[ox + 7] * kval;
                    }
                }
                continue;
            }
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
                int ox = 0;
                for (; ox < W; ox += 8) {
                    out_row[ox + 0] += in_row[ox + 0] * kval;
                    out_row[ox + 1] += in_row[ox + 1] * kval;
                    out_row[ox + 2] += in_row[ox + 2] * kval;
                    out_row[ox + 3] += in_row[ox + 3] * kval;
                    out_row[ox + 4] += in_row[ox + 4] * kval;
                    out_row[ox + 5] += in_row[ox + 5] * kval;
                    out_row[ox + 6] += in_row[ox + 6] * kval;
                    out_row[ox + 7] += in_row[ox + 7] * kval;
                }
            }
        }
    }
}
