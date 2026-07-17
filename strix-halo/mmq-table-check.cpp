// Standalone check for the RDNA3.5 MMQ config table (strix-halo Finding #9, see
// mmq-rdna3_5-config-table.md). Validates that mmq-config-rdna3_5.cuh is well-formed
// and dispatches as intended WITHOUT needing ROCm or gfx1151 hardware, so the table
// can be edited and swept from any dev host.
//
//   cd ggml/src/ggml-cuda
//   clang++ -std=c++17 -I. -Wall -o /tmp/mmq-table-check ../../../strix-halo/mmq-table-check.cpp
//   /tmp/mmq-table-check
//
// It stubs just enough of mmq.cuh to include the real config tables and reuses the
// real CASE macro verbatim, so the upstream static_asserts fire at compile time.
// This proves the table is CORRECT, not that it is FAST -- that still needs a bench.
//
// Keep the CASE macro and ggml_cuda_mmq_config below in sync with mmq.cuh if upstream
// changes them; a silent drift here turns this check into a rubber stamp.
#include <cstdio>
#include <set>
#include <tuple>

#define __host__
#define __device__
#define MMQ_ITER_K 256

enum ggml_type {
    GGML_TYPE_Q1_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
    GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, GGML_TYPE_COUNT,
};

enum ggml_cuda_mmq_sram_layout {
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4,
    GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4,
};

struct ggml_cuda_mmq_config {
    ggml_type type; int nthreads; int occupancy; int I; int J;
    ggml_cuda_mmq_sram_layout sram_layout; int K_vram; bool stream_k; bool fallback;
    constexpr __host__ __device__ ggml_cuda_mmq_config(
            ggml_type type, int nthreads, int occupancy, int I, int J,
            ggml_cuda_mmq_sram_layout sram_layout, int K_vram, bool stream_k, bool fallback) :
        type(type), nthreads(nthreads), occupancy(occupancy), I(I), J(J),
        sram_layout(sram_layout), K_vram(K_vram), stream_k(stream_k), fallback(fallback) {}
};

// Verbatim from mmq.cuh.
#define CASE(type_, nthreads_, occupancy_, I_, J_, sram_layout_, K_vram_, stream_k_, fallback_)                                           \
    if (type == (type_) && J == (J_) && fallback == (fallback_)) {                                                                        \
        static_assert((nthreads_) %  32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                \
        static_assert(                          (occupancy_)      <=   8, "bad occupancy");                                               \
        static_assert((I_)        %  32 == 0,                             "bad I");                                                       \
        static_assert((J_)        %   8 == 0,                             "bad J");                                                       \
        static_assert((K_vram_)   % 256 == 0,                             "bad K_vram");                                                  \
        return ggml_cuda_mmq_config((type_), (nthreads_), (occupancy_), (I_), (J_), (sram_layout_), (K_vram_), (stream_k_), (fallback_)); \
    }                                                                                                                                     \

#include "mmq-config-rdna3_5.cuh"
#include "mmq-config-rdna4.cuh"
#undef CASE

// Mirrors mul_mat_q_switch_J's selection loop (shared-memory filter omitted).
static int pick_J(ggml_type type, bool fallback, bool is_moe, bool rdna3_5, int ncols_max) {
    int J_best = 0, ntiles_J_best = 1 << 30;
    const int J_max = rdna3_5 && is_moe ? 48 : 128;
    for (int J = 8; J <= J_max && ntiles_J_best > 1; J += 8) {
        auto cfg = rdna3_5 ? ggml_cuda_mmq_get_config_rdna3_5(type, J, fallback)
                           : ggml_cuda_mmq_get_config_rdna4(type, J, fallback);
        if (cfg.type == GGML_TYPE_COUNT) continue;
        const int ntiles_x = (ncols_max + cfg.J - 1) / cfg.J;
        if (ntiles_x < ntiles_J_best) { J_best = J; ntiles_J_best = ntiles_x; }
    }
    return J_best;
}

int main() {
    int fails = 0;
    const ggml_type types[] = {
        GGML_TYPE_Q1_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
        GGML_TYPE_MXFP4, GGML_TYPE_NVFP4,
    };

    // 1. Every entry present in rdna4 must exist in rdna3_5 with nthreads=128/I=64,
    //    and identical sram_layout/K_vram/stream_k (only the tile shape was retuned).
    for (ggml_type t : types) {
        for (bool fb : {false, true}) {
            for (int J = 8; J <= 128; J += 8) {
                auto a = ggml_cuda_mmq_get_config_rdna4(t, J, fb);
                auto b = ggml_cuda_mmq_get_config_rdna3_5(t, J, fb);
                if (a.type == GGML_TYPE_COUNT) continue;
                if (b.type == GGML_TYPE_COUNT) {
                    printf("FAIL: rdna3_5 missing type=%d J=%d fallback=%d\n", (int)t, J, (int)fb);
                    fails++; continue;
                }
                if (b.nthreads != 128 || b.I != 64 || b.occupancy != 2) {
                    printf("FAIL: type=%d J=%d fb=%d -> nthreads=%d I=%d occ=%d\n",
                           (int)t, J, (int)fb, b.nthreads, b.I, b.occupancy);
                    fails++;
                }
                if (b.sram_layout != a.sram_layout || b.K_vram != a.K_vram || b.stream_k != a.stream_k) {
                    printf("FAIL: type=%d J=%d fb=%d diverged from rdna4 on layout/K_vram/stream_k\n",
                           (int)t, J, (int)fb);
                    fails++;
                }
            }
        }
    }

    // 2. I must equal nwarps*16 (wave32) -- the write-back invariant in mmq.cuh.
    for (ggml_type t : types) {
        auto b = ggml_cuda_mmq_get_config_rdna3_5(t, 64, false);
        if (b.type != GGML_TYPE_COUNT && b.I != (b.nthreads / 32) * 16) {
            printf("FAIL: I != nwarps*16 for type=%d (I=%d nthreads=%d)\n", (int)t, b.I, b.nthreads);
            fails++;
        }
    }

    // 3. J=48 must now exist for BOTH fallback values (the added rows).
    for (ggml_type t : types) {
        for (bool fb : {false, true}) {
            if (ggml_cuda_mmq_get_config_rdna3_5(t, 48, fb).type == GGML_TYPE_COUNT) {
                printf("FAIL: no J=48 entry for type=%d fallback=%d\n", (int)t, (int)fb);
                fails++;
            }
        }
    }

    // 4. MoE cap resolves to exactly 48 (not 32) for both fallback values; dense is unclamped.
    for (bool fb : {false, true}) {
        int moe   = pick_J(GGML_TYPE_Q4_K, fb, /*is_moe=*/true,  /*rdna3_5=*/true, 4096);
        int dense = pick_J(GGML_TYPE_Q4_K, fb, /*is_moe=*/false, /*rdna3_5=*/true, 4096);
        printf("Q4_K fallback=%d: MoE J=%d, dense J=%d\n", (int)fb, moe, dense);
        if (moe != 48)    { printf("FAIL: MoE J expected 48\n"); fails++; }
        if (dense != 128) { printf("FAIL: dense J expected 128\n"); fails++; }
    }

    // 5. Cap must never abort (J_best==0) for any type/fallback/ncols_max.
    for (ggml_type t : types) {
        for (bool fb : {false, true}) {
            for (int ncols : {1, 7, 8, 33, 512, 4096}) {
                if (pick_J(t, fb, true, true, ncols) == 0) {
                    printf("FAIL: J_best=0 (GGML_ABORT) type=%d fb=%d ncols=%d\n", (int)t, (int)fb, ncols);
                    fails++;
                }
            }
        }
    }

    printf(fails ? "\n%d FAILURES\n" : "\nall checks passed\n", fails);
    return fails != 0;
}
