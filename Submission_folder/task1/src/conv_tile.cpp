// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int TILE = 64;

    for (int i = 0; i < H * W; ++i) {
        out[i] = 0.0f;
    }
    for (int oy0 = 0; oy0 < H; oy0 += TILE) {
        for (int ox0 = 0; ox0 < W; ox0 += TILE) {

            const int oy_end = (oy0 + TILE < H) ? oy0 + TILE : H;
            const int ox_end = (ox0 + TILE < W) ? ox0 + TILE : W;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float kval = ker[ky * K + kx];

                    for (int oy = oy0; oy < oy_end; ++oy) {
                        const float* in_row =
                            in + (oy + ky) * in_stride + ox0 + kx;
                        float* out_row = out + oy * W + ox0;
                        for (int ox = 0; ox < ox_end - ox0; ++ox) {
                            out_row[ox] += in_row[ox] * kval;
                        }
                    }
                }
            }
        }
    }
}
