// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float kval = ker[ky * K + kx];
            if(kx==0 && ky==0){
                for (int oy = 0; oy < H; ++oy) {
                    const float* in_row = in + (oy + ky) * in_stride + kx;
                    float* out_row = out + oy * W;
                    for (int ox = 0; ox < W; ++ox) {
                        out_row[ox] = in_row[ox] * kval;
                    }
                }
                continue;
            }
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
                for (int ox = 0; ox < W; ++ox) {
                    out_row[ox] += in_row[ox] * kval;
                }
            }
        }
    }
}
