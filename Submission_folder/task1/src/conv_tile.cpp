#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int tile_y = 16; 
    const int tile_x = 256;

    for (int ty = 0; ty < H; ty += tile_y) {
        int max_oy = std::min(ty + tile_y, H);        
        for (int tx = 0; tx < W; tx += tile_x) {
            int max_ox = std::min(tx + tile_x, W);
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
