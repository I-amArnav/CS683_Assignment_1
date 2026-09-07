#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int TILE_Y = 64; 
    const int TILE_X = 64;

    for (int ty = 0; ty < H; ty += TILE_Y) {
        int max_oy = std::min(ty + TILE_Y, H);        
        for (int tx = 0; tx < W; tx += TILE_X) {
            int max_ox = std::min(tx + TILE_X, W);
            for (int oy = ty; oy < max_oy; ++oy) {
                for (int ox = tx; ox < max_ox; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}