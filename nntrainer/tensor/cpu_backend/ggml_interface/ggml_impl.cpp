#include "ggml-common.h"
#include "ggml-cpu-quants.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <ggml_interface.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <cstring>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif

typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
} q4_K_qparams;
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

static inline float nntr_compute_fp16_to_fp32(uint16_t h) {
  _FP16 tmp;
  memcpy(&tmp, &h, sizeof(uint16_t));
  return (float)tmp;
}

static inline uint16_t nntr_compute_fp32_to_fp16(float f) {
  uint16_t res;
  _FP16 tmp = f;
  memcpy(&res, &tmp, sizeof(uint16_t));
  return res;
}

static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

void __nntr_quantize_row_q8_0(const _FP16 *__restrict x, void *vy, int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0 *__restrict y = (block_q8_0 *__restrict)vy;

#if defined(__ARM_NEON)
  for (int i = 0; i < nb; i++) {
    float16x8_t srcv[4];  // loaded source
    float16x8_t asrcv[4]; // absolute value of source
    float16x8_t amaxv[2]; // absolute max buffer

    for (int j = 0; j < 4; j++) {
      srcv[j] = vld1q_f16(x + i * 32 + 8 * j);
    }
    for (int j = 0; j < 4; j++) {
      asrcv[j] = vabsq_f16(srcv[j]);
    }

    for (int j = 0; j < 2; j++) {
      amaxv[j] =
        vmaxq_f16(asrcv[2 * j], asrcv[2 * j + 1]); // 0, 1 <- 0, 1 VS 2, 3
    }
    amaxv[0] = vmaxq_f16(amaxv[0], amaxv[1]); // 0 <- 0, 1

    const float amax = static_cast<float>(vmaxvq_f16(amaxv[0]));

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_compute_fp32_to_fp16(d);

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
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = std::max(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = nntr_compute_fp32_to_fp16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = std::roundf(x0);
    }
  }
#endif
}

void __nntr_quantize_row_q8_0_ref_lossless(const float * GGML_RESTRICT x, void * GGML_RESTRICT _y, int64_t k, void * GGML_RESTRICT _y_ref) {
    
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y_ref = (block_q8_0 * GGML_RESTRICT)_y_ref;
    block_q8_0 * GGML_RESTRICT y =  (block_q8_0 * GGML_RESTRICT) _y;

    for (int i = 0; i < nb; i++) {
        const float d = nntr_compute_fp16_to_fp32(y_ref[i].d);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = y_ref[i].d;

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}

void __nntr_quantize_row_q4_K_ref_lossless(const float * GGML_RESTRICT x, void * GGML_RESTRICT _y, int64_t k, void * GGML_RESTRICT _y_ref) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q4_K * GGML_RESTRICT y =  (block_q4_K * GGML_RESTRICT) _y;
    block_q4_K * GGML_RESTRICT y_ref =  (block_q4_K * GGML_RESTRICT) _y_ref;

    uint8_t L[QK_K];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K/32; ++j) {
            if (j < 4) {
                y[i].scales[j] = y_ref[i].scales[j];
                y[i].scales[j+4] = y_ref[i].scales[j+4];
            } else {
                y[i].scales[j+4] = y_ref[i].scales[j+4];
                y[i].scales[j-4] |= y_ref[i].scales[j-4];
                y[i].scales[j-0] |= y_ref[i].scales[j-0];
            }
        }
        y[i].d = y_ref[i].d ;
        y[i].dmin = y_ref[i].dmin;

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = nntr_compute_fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            const float dm = nntr_compute_fp16_to_fp32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }
        x += QK_K;
    }
}

size_t __nntr_quantize_q8_0(const _FP16 *src, void *dst, int64_t nrow,
                            int64_t n_per_row, const float *quant_weights) {
  const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
  __nntr_quantize_row_q8_0(src, dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

void __nntr_dequantize_row_q8_0(const void *_x, _FP16 *__restrict y,
                                int64_t k) {
  static const int qk = QK8_0;
  const block_q8_0 *__restrict x = (const block_q8_0 *__restrict)_x;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    // const _FP16 d = (x[i].d); ///@todo check if this works
    const float d = nntr_compute_fp16_to_fp32(x[i].d);

    for (int j = 0; j < qk; ++j) {
      y[i * qk + j] = x[i].qs[j] * d;
    }
  }
}

#if 0
void ggml_quantize_mat_q8_0_4x8(const _FP16 * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float16x8_t srcv[4][4];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float16x8_t asrcv[4];
        float16x8_t amaxv[2];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 4; j++) srcv[row_iter][j] = vld1q_f16(x + row_iter * k + i * 32 + 8 * j);
            for (int j = 0; j < 4; j++) asrcv[j] = vabsq_f16(srcv[row_iter][j]);

            for (int j = 0; j < 2; j++) {
              amaxv[j] =
                vmaxq_f16(asrcv[2 * j], asrcv[2 * j + 1]); // 0, 1 <- 0, 1 VS 2, 3
            }
            amaxv[0] = vmaxq_f16(amaxv[0], amaxv[1]); // 0 <- 0, 1

            const float amax = vmaxvq_f16(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
        }

        for (int j = 0; j < 2; j++) {
            float16x8_t v = vmulq_n_f16(srcv[0][2 * j], id[0]);
            int16x8_t vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 0] = vgetq_lane_s16(vi, 0);
            y[i].qs[32 * j + 1] = vgetq_lane_s16(vi, 1);
            y[i].qs[32 * j + 2] = vgetq_lane_s16(vi, 2);
            y[i].qs[32 * j + 3] = vgetq_lane_s16(vi, 3);

            y[i].qs[32 * j + 4] = vgetq_lane_s16(vi, 4);
            y[i].qs[32 * j + 5] = vgetq_lane_s16(vi, 5);
            y[i].qs[32 * j + 6] = vgetq_lane_s16(vi, 6);
            y[i].qs[32 * j + 7] = vgetq_lane_s16(vi, 7);
            
            v = vmulq_n_f16(srcv[0][2 * j + 1], id[0]);
            vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f16(srcv[1][2 * j], id[1]);
            vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f16(srcv[1][2 * j + 1], id[1]);
            vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f16(srcv[2][2 * j], id[2]);
            vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f16(srcv[2][2 * j + 1], id[2]);
            vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f16(srcv[3][2 * j], id[3]);
            vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f16(srcv[3][2 * j + 1], id[3]);
            vi = vcvtnq_s16_f16(v);
            y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}
#endif

} // namespace nntrainer
