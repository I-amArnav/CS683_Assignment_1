#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int TILE_H = 16;
    const int TILE_W = 256;

    for (int oy0 = 0; oy0 < H; oy0 += TILE_H) {
        const int oy_end = (oy0 + TILE_H < H) ? oy0 + TILE_H : H;

        for (int ox0 = 0; ox0 < W; ox0 += TILE_W) {
            const int ox_end = (ox0 + TILE_W < W) ? ox0 + TILE_W : W;

            for (int oy = oy0; oy < oy_end; ++oy) {
                float* out_row = out + oy * W + ox0;
                for (int ox = ox0; ox < ox_end; ++ox) {
                    out_row[ox - ox0] = 0.0f;
                }
            }
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float kval = ker[ky * K + kx];
                    for (int oy = oy0; oy < oy_end; ++oy) {
                        const float* in_row = in + (oy + ky) * in_stride + ox0 + kx;
                        float* out_row = out + oy * W + ox0;
                        for (int ox = ox0; ox < ox_end; ++ox) {
                            out_row[ox - ox0] += in_row[ox - ox0] * kval;
                        }
                    }
                }
            }
        }
    }
}
