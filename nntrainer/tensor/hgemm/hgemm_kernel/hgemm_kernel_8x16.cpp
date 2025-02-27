// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x16.cpp
 * @date   04 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x16 kernel
 *
 */

#include <arm_neon.h>
#include <assert.h>
#include <hgemm_kernel.h>
#include <hgemm_util.h>
#include <stdlib.h>

#define INIT_KERNEL_8X16()       \
  do {                           \
    v0_7 = vdupq_n_f16(0.F);     \
    v8_15 = vdupq_n_f16(0.F);    \
    v16_23 = vdupq_n_f16(0.F);   \
    v24_31 = vdupq_n_f16(0.F);   \
    v32_39 = vdupq_n_f16(0.F);   \
    v40_47 = vdupq_n_f16(0.F);   \
    v48_55 = vdupq_n_f16(0.F);   \
    v56_63 = vdupq_n_f16(0.F);   \
    v64_71 = vdupq_n_f16(0.F);   \
    v72_79 = vdupq_n_f16(0.F);   \
    v80_87 = vdupq_n_f16(0.F);   \
    v88_95 = vdupq_n_f16(0.F);   \
    v96_103 = vdupq_n_f16(0.F);  \
    v104_111 = vdupq_n_f16(0.F); \
    v112_119 = vdupq_n_f16(0.F); \
    v120_127 = vdupq_n_f16(0.F); \
  } while (0)

#define KERNEL_8x16_ACC_N4(N)                            \
  do {                                                   \
    for (int i = 0; i < N; i += 4) {                     \
      va0 = vld1q_f16(a + 8 * i);                        \
      vb1 = vld1q_f16(b + 16 * i);                       \
      vb2 = vld1q_f16(b + 16 * i + 8 * 1);               \
      v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
      v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
      v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
      v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
      v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
      v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
      v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
      v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
      v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
      v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
      v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
      v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
      v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
      v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
      v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
      v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
      va0 = vld1q_f16(a + 8 * i + 8 * 1);                \
      vb1 = vld1q_f16(b + 16 * i + 8 * 2);               \
      vb2 = vld1q_f16(b + 16 * i + 8 * 3);               \
      v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
      v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
      v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
      v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
      v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
      v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
      v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
      v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
      v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
      v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
      v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
      v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
      v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
      v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
      v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
      v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
      va0 = vld1q_f16(a + 8 * i + 8 * 2);                \
      vb1 = vld1q_f16(b + 16 * i + 8 * 4);               \
      vb2 = vld1q_f16(b + 16 * i + 8 * 5);               \
      v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
      v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
      v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
      v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
      v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
      v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
      v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
      v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
      v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
      v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
      v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
      v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
      v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
      v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
      v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
      v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
      va0 = vld1q_f16(a + 8 * i + 8 * 3);                \
      vb1 = vld1q_f16(b + 16 * i + 8 * 6);               \
      vb2 = vld1q_f16(b + 16 * i + 8 * 7);               \
      v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
      v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
      v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
      v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
      v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
      v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
      v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
      v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
      v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
      v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
      v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
      v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
      v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
      v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
      v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
      v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    }                                                    \
    l += N;                                              \
    __builtin_prefetch(b + 16 * N, 0, 3);                \
    __builtin_prefetch(a + 8 * N, 0, 3);                 \
    b += 16 * N;                                         \
    a += 8 * N;                                          \
  } while (0)

#define KERNEL_8x16_ACC1()                             \
  do {                                                 \
    va0 = vld1q_f16(a);                                \
    vb1 = vld1q_f16(b);                                \
    vb2 = vld1q_f16(b + 8);                            \
    v0_7 = vfmaq_laneq_f16(v0_7, vb1, va0, 0);         \
    v8_15 = vfmaq_laneq_f16(v8_15, vb1, va0, 1);       \
    v16_23 = vfmaq_laneq_f16(v16_23, vb1, va0, 2);     \
    v24_31 = vfmaq_laneq_f16(v24_31, vb1, va0, 3);     \
    v32_39 = vfmaq_laneq_f16(v32_39, vb1, va0, 4);     \
    v40_47 = vfmaq_laneq_f16(v40_47, vb1, va0, 5);     \
    v48_55 = vfmaq_laneq_f16(v48_55, vb1, va0, 6);     \
    v56_63 = vfmaq_laneq_f16(v56_63, vb1, va0, 7);     \
    v64_71 = vfmaq_laneq_f16(v64_71, vb2, va0, 0);     \
    v72_79 = vfmaq_laneq_f16(v72_79, vb2, va0, 1);     \
    v80_87 = vfmaq_laneq_f16(v80_87, vb2, va0, 2);     \
    v88_95 = vfmaq_laneq_f16(v88_95, vb2, va0, 3);     \
    v96_103 = vfmaq_laneq_f16(v96_103, vb2, va0, 4);   \
    v104_111 = vfmaq_laneq_f16(v104_111, vb2, va0, 5); \
    v112_119 = vfmaq_laneq_f16(v112_119, vb2, va0, 6); \
    v120_127 = vfmaq_laneq_f16(v120_127, vb2, va0, 7); \
    l += 1;                                            \
    __builtin_prefetch(b + 16, 0, 3);                  \
    __builtin_prefetch(a + 8, 0, 3);                   \
    b += 16 * 1;                                       \
    a += 8 * 1;                                        \
  } while (0)

#define SAVE_KERNEL_8X16_F16_F32()                                             \
  do {                                                                         \
    vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0_7))));   \
    vst1q_f32(c + 4,                                                           \
              vaddq_f32(vld1q_f32(c + 4), vcvt_f32_f16(vget_high_f16(v0_7)))); \
                                                                               \
    vst1q_f32(                                                                 \
      c + 8, vaddq_f32(vld1q_f32(c + 8), vcvt_f32_f16(vget_low_f16(v64_71)))); \
    vst1q_f32(c + 8 + 4, vaddq_f32(vld1q_f32(c + 8 + 4),                       \
                                   vcvt_f32_f16(vget_high_f16(v64_71))));      \
                                                                               \
    vst1q_f32(c + ldc, vaddq_f32(vld1q_f32(c + ldc),                           \
                                 vcvt_f32_f16(vget_low_f16(v8_15))));          \
    vst1q_f32(c + ldc + 4, vaddq_f32(vld1q_f32(c + ldc + 4),                   \
                                     vcvt_f32_f16(vget_high_f16(v8_15))));     \
                                                                               \
    vst1q_f32(c + ldc + 8, vaddq_f32(vld1q_f32(c + ldc + 8),                   \
                                     vcvt_f32_f16(vget_low_f16(v72_79))));     \
    vst1q_f32(c + ldc + 8 + 4,                                                 \
              vaddq_f32(vld1q_f32(c + ldc + 8 + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v72_79))));                 \
                                                                               \
    vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v16_23))));     \
    vst1q_f32(c + 2 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 2 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v16_23))));                 \
                                                                               \
    vst1q_f32(c + 2 * ldc + 8, vaddq_f32(vld1q_f32(c + 2 * ldc + 8),           \
                                         vcvt_f32_f16(vget_low_f16(v80_87)))); \
    vst1q_f32(c + 2 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 2 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v80_87))));                 \
                                                                               \
    vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v24_31))));     \
    vst1q_f32(c + 3 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 3 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v24_31))));                 \
                                                                               \
    vst1q_f32(c + 3 * ldc + 8, vaddq_f32(vld1q_f32(c + 3 * ldc + 8),           \
                                         vcvt_f32_f16(vget_low_f16(v88_95)))); \
    vst1q_f32(c + 3 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 3 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v88_95))));                 \
                                                                               \
    vst1q_f32(c + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v32_39))));     \
    vst1q_f32(c + 4 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 4 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v32_39))));                 \
                                                                               \
    vst1q_f32(c + 4 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 4 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v96_103))));                 \
    vst1q_f32(c + 4 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 4 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v96_103))));                \
                                                                               \
    vst1q_f32(c + 5 * ldc, vaddq_f32(vld1q_f32(c + 5 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v40_47))));     \
    vst1q_f32(c + 5 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 5 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v40_47))));                 \
    vst1q_f32(c + 5 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 5 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v104_111))));                \
    vst1q_f32(c + 5 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 5 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v104_111))));               \
                                                                               \
    vst1q_f32(c + 6 * ldc, vaddq_f32(vld1q_f32(c + 6 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v48_55))));     \
    vst1q_f32(c + 6 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 6 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v48_55))));                 \
                                                                               \
    vst1q_f32(c + 6 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 6 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v112_119))));                \
    vst1q_f32(c + 6 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 6 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v112_119))));               \
                                                                               \
    vst1q_f32(c + 7 * ldc, vaddq_f32(vld1q_f32(c + 7 * ldc),                   \
                                     vcvt_f32_f16(vget_low_f16(v56_63))));     \
    vst1q_f32(c + 7 * ldc + 4,                                                 \
              vaddq_f32(vld1q_f32(c + 7 * ldc + 4),                            \
                        vcvt_f32_f16(vget_high_f16(v56_63))));                 \
                                                                               \
    vst1q_f32(c + 7 * ldc + 8,                                                 \
              vaddq_f32(vld1q_f32(c + 7 * ldc + 8),                            \
                        vcvt_f32_f16(vget_low_f16(v120_127))));                \
    vst1q_f32(c + 7 * ldc + 8 + 4,                                             \
              vaddq_f32(vld1q_f32(c + 7 * ldc + 8 + 4),                        \
                        vcvt_f32_f16(vget_high_f16(v120_127))));               \
  } while (0)

template <>
void hgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
                       __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 16 == 0 && K % 8 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      // 8x16
      float16x8_t v0_7, v8_15;
      float16x8_t v16_23, v24_31;
      float16x8_t v32_39, v40_47;
      float16x8_t v48_55, v56_63;
      float16x8_t v64_71, v72_79;
      float16x8_t v80_87, v88_95;
      float16x8_t v96_103, v104_111;
      float16x8_t v112_119, v120_127;
      float16x8_t vb1, vb2;
      float16x8_t va0;

      INIT_KERNEL_8X16();
      l = 0;
      for (; l < K;) {
        KERNEL_8x16_ACC1();
      }
      vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0_7));
      vst1q_f16(c + 8, vaddq_f16(vld1q_f16(c + 8), v64_71));
      vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v8_15));
      vst1q_f16(c + ldc + 8, vaddq_f16(vld1q_f16(c + ldc + 8), v72_79));
      vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v16_23));
      vst1q_f16(c + 2 * ldc + 8, vaddq_f16(vld1q_f16(c + 2 * ldc + 8), v80_87));
      vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v24_31));
      vst1q_f16(c + 3 * ldc + 8, vaddq_f16(vld1q_f16(c + 3 * ldc + 8), v88_95));
      vst1q_f16(c + 4 * ldc, vaddq_f16(vld1q_f16(c + 4 * ldc), v32_39));
      vst1q_f16(c + 4 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 4 * ldc + 8), v96_103));
      vst1q_f16(c + 5 * ldc, vaddq_f16(vld1q_f16(c + 5 * ldc), v40_47));
      vst1q_f16(c + 5 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 5 * ldc + 8), v104_111));
      vst1q_f16(c + 6 * ldc, vaddq_f16(vld1q_f16(c + 6 * ldc), v48_55));
      vst1q_f16(c + 6 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 6 * ldc + 8), v112_119));
      vst1q_f16(c + 7 * ldc, vaddq_f16(vld1q_f16(c + 7 * ldc), v56_63));
      vst1q_f16(c + 7 * ldc + 8,
                vaddq_f16(vld1q_f16(c + 7 * ldc + 8), v120_127));
      c += 16;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}

// template <>
// void hgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
//                        __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
//   assert(M > 0 && N > 0 && K > 0);
//   assert(M % 8 == 0 && N % 16 == 0 && K % 4 == 0);

//   __fp16 *a = sa, *b = sb;
//   float *c = sc;
//   unsigned int i, j, l;
//   unsigned int K4 = get_prev_mltpl_of_2p_n(K, 2);
//   unsigned int K8 = get_prev_mltpl_of_2p_n(K, 3);
//   unsigned int K16 = get_prev_mltpl_of_2p_n(K, 4);
//   for (i = 0; i < M; i += 8) {
//     for (j = 0; j < N; j += 16) {
//       __builtin_prefetch(b, 0, 3);
//       __builtin_prefetch(a, 0, 3);
//       float16x8_t v0_7, v8_15;
//       float16x8_t v16_23, v24_31;
//       float16x8_t v32_39, v40_47;
//       float16x8_t v48_55, v56_63;
//       float16x8_t v64_71, v72_79;
//       float16x8_t v80_87, v88_95;
//       float16x8_t v96_103, v104_111;
//       float16x8_t v112_119, v120_127;
//       float16x8_t va0;
//       float16x8_t vb1, vb2;
//       l = 0;
//       for (; l < K16;) {
//         INIT_KERNEL_8X16();
//         KERNEL_8x16_ACC_N4(16);
//         SAVE_KERNEL_8X16_F16_F32();
//       }
//       for (; l < K8;) {
//         INIT_KERNEL_8X16();
//         KERNEL_8x16_ACC_N4(8);
//         SAVE_KERNEL_8X16_F16_F32();
//       }
//       for (; l < K4;) {
//         INIT_KERNEL_8X16();
//         KERNEL_8x16_ACC_N4(4);
//         SAVE_KERNEL_8X16_F16_F32();
//       }
//       for (; l < K;) {
//         INIT_KERNEL_8X16();
//         KERNEL_8x16_ACC1();
//         SAVE_KERNEL_8X16_F16_F32();
//       }
//       c += 16;
//       a -= 8 * K;
//     }
//     sc += ldc * 8;
//     c = sc;
//     a += 8 * K;
//     b = sb;
//   }
// }

// zero-init like this works!
#define HGEMM_UKERNEL_8x16_INIT_ASM \
  "eor v6.16b,  v6.16b,  v6.16b \n\t"            \
  "eor v7.16b, v7.16b, v7.16b \n\t"            \
  "eor v8.16b, v8.16b, v8.16b \n\t"            \
  "eor v9.16b, v9.16b, v9.16b \n\t"            \
  "eor v10.16b, v10.16b, v10.16b \n\t"           \
  "eor v11.16b, v11.16b, v11.16b \n\t"           \
  "eor v12.16b, v12.16b, v12.16b \n\t"           \
  "eor v13.16b, v13.16b, v13.16b \n\t"           \
  "eor v14.16b, v14.16b, v14.16b \n\t"           \
  "eor v15.16b, v15.16b, v15.16b \n\t"           \
  "eor v16.16b, v16.16b, v16.16b \n\t"           \
  "eor v17.16b, v17.16b, v17.16b \n\t"           \
  "eor v18.16b, v18.16b, v18.16b \n\t"           \
  "eor v19.16b, v19.16b, v19.16b \n\t"           \
  "eor v20.16b, v20.16b, v20.16b \n\t"           \
  "eor v21.16b, v21.16b, v21.16b \n\t"

/**
 * @note v0 : a1, v1 : b1, v2 : b2, v3 ~ v5 : ukernel, v6 ~ v30 : intermediate
 * registers
 *
 */
#define HGEMM_KERNEL_8x16_ukernel_ASM \
  "mov w0, %w[acc_n] \n\t"            \
  "1:                 \n\t"                           \
  "ld1 {v3.8h}, [%[a]], #16 \n\t"       \
  "ld1 v4.8h, [%[b]], #16 \n\t"       \
  "ld1 v5.8h, [%[b]], #16 \n\t"       \
  "fmla v6.8h, v4.8h, v3.h[0] \n\t"   \
  "fmla v7.8h, v4.8h, v3.h[1] \n\t"   \
  "fmla v8.8h, v4.8h, v3.h[2] \n\t"   \
  "fmla v9.8h, v4.8h, v3.h[3] \n\t"   \
  "fmla v10.8h, v4.8h, v3.h[4] \n\t"  \
  "fmla v11.8h, v4.8h, v3.h[5] \n\t"  \
  "fmla v12.8h, v4.8h, v3.h[6] \n\t"  \
  "fmla v13.8h, v4.8h, v3.h[7] \n\t"  \
  "fmla v14.8h, v5.8h, v3.h[0] \n\t"  \
  "fmla v15.8h, v5.8h, v3.h[1] \n\t"  \
  "fmla v16.8h, v5.8h, v3.h[2] \n\t"  \
  "fmla v17.8h, v5.8h, v3.h[3] \n\t"  \
  "fmla v18.8h, v5.8h, v3.h[4] \n\t"  \
  "fmla v19.8h, v5.8h, v3.h[5] \n\t"  \
  "fmla v20.8h, v5.8h, v3.h[6] \n\t"  \
  "fmla v21.8h, v5.8h, v3.h[7] \n\t"  \
  "subs w0, w0, #1 \n\t"              \
  "b.ne 1b \n\t"                      \
  "2: \n\t"

#define HGEMM_KERNEL_8x16_SAVE_ASM                                            \
  "lsl x10, %[ldc], #2 \n\t" /* --- Row 0 (pointer = c) --- */                \
  "ld1 {v22.4s}, [%[c]] \n\t"                                                 \
  "fcvtl v23.4s, v6.4h \n\t" /* convert lower half */                         \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [%[c]] \n\t" /* Group 1: columns 4-7 from v6 (upper half) */ \
  "ld1 {v22.4s}, [%[c], #16] \n\t"                                            \
  "fcvtl2 v23.4s, v6.8h \n\t" /* convert upper half */                        \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [%[c], #16] \n\t"                                            \
  "ld1 {v22.4s}, [%[c], #32] \n\t"                                            \
  "fcvtl v23.4s, v14.4h \n\t"                                                 \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [%[c], #32] \n\t"                                            \
  "ld1 {v22.4s}, [%[c], #48] \n\t"                                            \
  "fcvtl2 v23.4s, v14.8h \n\t"                                                \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [%[c], #48] \n\t" /* --- Row 1 (pointer = c + x10) --- */    \
  "add x11, %[c], x10 \n\t"        /* x11 = row1 pointer */                   \
  "ld1 {v22.4s}, [x11] \n\t"                                                  \
  "fcvtl v23.4s, v7.4h \n\t"                                                  \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11] \n\t"                                                  \
  "ld1 {v22.4s}, [x11, #16] \n\t"                                             \
  "fcvtl2 v23.4s, v7.8h \n\t"                                                 \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11, #16] \n\t"                                             \
  "ld1 {v22.4s}, [x11, #32] \n\t"                                             \
  "fcvtl v23.4s, v15.4h \n\t"                                                 \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11, #32] \n\t"                                             \
  "ld1 {v22.4s}, [x11, #48] \n\t"                                             \
  "fcvtl2 v23.4s, v15.8h \n\t"                                                \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11, #48] \n\t" /* --- Row 2 (pointer = x11 + x10) --- */   \
  "add x11, x11, x10 \n\t"        /* now row2 */                              \
  "ld1 {v22.4s}, [x11] \n\t"                                                  \
  "fcvtl v23.4s, v8.4h \n\t"                                                  \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11] \n\t"                                                  \
  "ld1 {v22.4s}, [x11, #16] \n\t"                                             \
  "fcvtl2 v23.4s, v8.8h \n\t"                                                 \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11, #16] \n\t"                                             \
  "ld1 {v22.4s}, [x11, #32] \n\t"                                             \
  "fcvtl v23.4s, v16.4h \n\t"                                                 \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11, #32] \n\t"                                             \
  "ld1 {v22.4s}, [x11, #48] \n\t"                                             \
  "fcvtl2 v23.4s, v16.8h \n\t"                                                \
  "fadd v22.4s, v22.4s, v23.4s \n\t"                                          \
  "st1 {v22.4s}, [x11, #48] \n\t"

#define HGEMM_KERNEL_8x16_ACC_N_ASM()                                         \
  do {                                                                        \
    __asm__ volatile(                                                         \
      HGEMM_UKERNEL_8x16_INIT_ASM HGEMM_KERNEL_8x16_ukernel_ASM               \
        HGEMM_KERNEL_8x16_SAVE_ASM                                            \
      : [a] "+x"(a), [b] "+x"(b), [c] "+x"(c), [l] "+r"(l), [ldc] "+r"(ldc)   \
      : [acc_n] "+r"(acc_n)                                                   \
      : "w0", "v3", "v4", "v5", "v6", "v7", "v8", "v9",     \
        "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", \
        "v20", "v21", "v22", "v23", "v24", "x10", "x11", "memory");           \
  } while (0)

template <>
void hgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
                       __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 16 == 0 && K % 4 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  unsigned int K4 = get_prev_mltpl_of_2p_n(K, 2);
  unsigned int K8 = get_prev_mltpl_of_2p_n(K, 3);
  unsigned int K16 = get_prev_mltpl_of_2p_n(K, 4);
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      l = 0;
      unsigned int acc_n = 4;
      __asm__ volatile(
        HGEMM_UKERNEL_8x16_INIT_ASM HGEMM_KERNEL_8x16_ukernel_ASM
          HGEMM_KERNEL_8x16_SAVE_ASM
        : [a] "+x"(a), [b] "+x"(b), [c] "+x"(c), [l] "+r"(l), [ldc] "+r"(ldc)
        : [acc_n] "+r"(acc_n)
        : "w0", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "x10", "x11", "memory");
      c += 16;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}
