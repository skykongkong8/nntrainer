#include "ggml-common.h"
#include "ggml-cpu-quants.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <ggml_interface.h>
#include <math.h>
#include <stdint.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
namespace nntrainer {
template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return QK4_0;
  }
  if constexpr (K == 8) {
    return QK8_0;
  }
  return -1;
}

template <int K, int N> struct block {
  uint16_t d[N];                      // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q8_0x4 = block<8, 4>;

void __ggml_quantize_row_q8_0(const _FP16 *__restrict x, void *vy, int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0 *__restrict y = (block_q8_0 *__restrict)vy;

#if defined(__ARM_NEON)
  for (int i = 0; i < nb; i++) {
    float16x8_t srcv[4];  // loaded source
    float16x8_t asrcv[4]; // absolute value of source
    float16x8_t amaxv[4];

    for (int j = 0; j < 4; j++)
      srcv[j] = vld1q_f16(x + i * 32 + 8 * j);
    for (int j = 0; j < 4; j++)
      asrcv[j] = vabsq_f16(srcv[j]);

    for (int j = 0; j < 2; j++)
      amaxv[2 * j] =
        vmaxq_f16(amaxv[2 * j], amaxv[2 * j + 1]); // 0 2 <- 0 2 VS 1 3
    for (int j = 0; j < 1; j++)
      amaxv[4 * j] = vmaxq_f16(amaxv[4 * j], amaxv[4 * j + 2]); // 0 <- 0 VS 2

    const float amax = static_cast<float>(
      vmaxvq_f16(amaxv[0])); // TODO : test this for d and id values
    // const _FP16 amax = vmaxvq_f16(amaxv[0]);

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = static_cast<float>(d); ///@todo : check if this works
    // y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < 4; j++) {
      const float16x8_t v = vmulq_n_f16(srcv[j], id);
      const int16x8_t vi = vcvtnq_s16_f16(v);

      y[i].qs[8 * j + 0] = vgetq_lane_s16(vi, 0);
      y[i].qs[8 * j + 1] = vgetq_lane_s16(vi, 1);
      y[i].qs[8 * j + 2] = vgetq_lane_s16(vi, 2);
      y[i].qs[8 * j + 3] = vgetq_lane_s16(vi, 3);
      y[i].qs[8 * j + 4] = vgetq_lane_s16(vi, 4);
      y[i].qs[8 * j + 5] = vgetq_lane_s16(vi, 5);
      y[i].qs[8 * j + 6] = vgetq_lane_s16(vi, 6);
      y[i].qs[8 * j + 7] = vgetq_lane_s16(vi, 7);
    }
  }
#else
  ///@todo convert to F16 version
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = std::max(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = static_cast<float>(d); ///@todo check if this works
    // y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = std::roundf(x0);
    }
  }
#endif
}

size_t __ggml_quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
  __ggml_quantize_row_q8_0(src, dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

void __ggml_dequantize_row_q8_0(const void *_x, _FP16 *__restrict y,
                                int64_t k) {
  static const int qk = QK8_0;
  const block_q8_0 *__restrict x = (const block_q8_0 *__restrict)_x;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    const _FP16 d = (x[i].d); ///@todo check if this works
    // const float d = GGML_FP16_TO_FP32(x[i].d);

    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = x[i].qs[j] * d;
    }
  }
}

#if 0
// void ggml_quantize_mat_q8_0_4x8(const _FP16 * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
//     assert(QK8_0 == 32);
//     assert(k % QK8_0 == 0);
//     const int nb = k / QK8_0;

//     block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

// #if defined(__ARM_NEON)
//     float32x4_t srcv[4][8];
//     float id[4];

//     for (int i = 0; i < nb; i++) {
//         float32x4_t asrcv[8];
//         float32x4_t amaxv[8];

//         for (int row_iter = 0; row_iter < 4; row_iter++) {
//             for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
//             for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

//             for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
//             for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
//             for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

//             const float amax = vmaxvq_f32(amaxv[0]);

//             const float d = amax / ((1 << 7) - 1);
//             id[row_iter] = d ? 1.0f / d : 0.0f;

//             y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
//         }

//         for (int j = 0; j < 4; j++) {
//             float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
//             int32x4_t vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
//             v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
//             vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

//             v = vmulq_n_f32(srcv[1][2 * j], id[1]);
//             vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
//             v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
//             vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

//             v = vmulq_n_f32(srcv[2][2 * j], id[2]);
//             vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
//             v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
//             vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

//             v = vmulq_n_f32(srcv[3][2 * j], id[3]);
//             vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
//             v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
//             vi = vcvtnq_s32_f32(v);
//             y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
//             y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
//             y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
//             y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
//         }
//     }
// #else
//     // scalar
//     const int blck_size_interleave = 8;
//     float srcv[4][QK8_0];
//     float id[4];

//     for (int i = 0; i < nb; i++) {
//         for (int row_iter = 0; row_iter < 4; row_iter++) {
//             float amax = 0.0f; // absolute max

//             for (int j = 0; j < QK8_0; j++) {
//                 srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
//                 amax = MAX(amax, fabsf(srcv[row_iter][j]));
//             }

//             const float d = amax / ((1 << 7) - 1);
//             id[row_iter] = d ? 1.0f / d : 0.0f;

//             y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
//         }

//         for (int j = 0; j < QK8_0 * 4; j++) {
//             int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
//             int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
//             src_offset += (j % blck_size_interleave);

//             float x0 = srcv[src_id][src_offset] * id[src_id];
//             y[i].qs[j] = roundf(x0);
//         }
//     }
// #endif
// }
#endif

} // namespace nntrainer
