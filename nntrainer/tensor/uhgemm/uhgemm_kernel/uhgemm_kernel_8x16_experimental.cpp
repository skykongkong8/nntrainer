// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_kernel_8x16_experimental.cpp
 * @date   04 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x16 experimental kernel
 *
 */

#include <arm_neon.h>
#include <assert.h>
#include <uhgemm_kernel.h>
#include <uhgemm_util.h>
#include <stdexcept>
#include <stdlib.h>

#define INIT_KERNEL_8X16()       \
  do {                           \
    v0_7 = vdupq_n_u16(0);     \
    v8_15 = vdupq_n_u16(0);    \
    v16_23 = vdupq_n_u16(0);   \
    v24_31 = vdupq_n_u16(0);   \
    v32_39 = vdupq_n_u16(0);   \
    v40_47 = vdupq_n_u16(0);   \
    v48_55 = vdupq_n_u16(0);   \
    v56_63 = vdupq_n_u16(0);   \
    v64_71 = vdupq_n_u16(0);   \
    v72_79 = vdupq_n_u16(0);   \
    v80_87 = vdupq_n_u16(0);   \
    v88_95 = vdupq_n_u16(0);   \
    v96_103 = vdupq_n_u16(0);  \
    v104_111 = vdupq_n_u16(0); \
    v112_119 = vdupq_n_u16(0); \
    v120_127 = vdupq_n_u16(0); \
  } while (0)

#define KERNEL_8x16_ACC_N4(N)                            \
  do {                                                   \
    for (int i = 0; i < N; i += 4) {                     \
      va0 = vld1q_u16(a + 8 * i);                        \
      vb1 = vld1q_u16(b + 16 * i);                       \
      vb2 = vld1q_u16(b + 16 * i + 8 * 1);               \
      v0_7 = vmlaq_laneq_u16(v0_7, vb1, va0, 0);         \
      v8_15 = vmlaq_laneq_u16(v8_15, vb1, va0, 1);       \
      v16_23 = vmlaq_laneq_u16(v16_23, vb1, va0, 2);     \
      v24_31 = vmlaq_laneq_u16(v24_31, vb1, va0, 3);     \
      v32_39 = vmlaq_laneq_u16(v32_39, vb1, va0, 4);     \
      v40_47 = vmlaq_laneq_u16(v40_47, vb1, va0, 5);     \
      v48_55 = vmlaq_laneq_u16(v48_55, vb1, va0, 6);     \
      v56_63 = vmlaq_laneq_u16(v56_63, vb1, va0, 7);     \
      v64_71 = vmlaq_laneq_u16(v64_71, vb2, va0, 0);     \
      v72_79 = vmlaq_laneq_u16(v72_79, vb2, va0, 1);     \
      v80_87 = vmlaq_laneq_u16(v80_87, vb2, va0, 2);     \
      v88_95 = vmlaq_laneq_u16(v88_95, vb2, va0, 3);     \
      v96_103 = vmlaq_laneq_u16(v96_103, vb2, va0, 4);   \
      v104_111 = vmlaq_laneq_u16(v104_111, vb2, va0, 5); \
      v112_119 = vmlaq_laneq_u16(v112_119, vb2, va0, 6); \
      v120_127 = vmlaq_laneq_u16(v120_127, vb2, va0, 7); \
      va0 = vld1q_u16(a + 8 * i + 8 * 1);                \
      vb1 = vld1q_u16(b + 16 * i + 8 * 2);               \
      vb2 = vld1q_u16(b + 16 * i + 8 * 3);               \
      v0_7 = vmlaq_laneq_u16(v0_7, vb1, va0, 0);         \
      v8_15 = vmlaq_laneq_u16(v8_15, vb1, va0, 1);       \
      v16_23 = vmlaq_laneq_u16(v16_23, vb1, va0, 2);     \
      v24_31 = vmlaq_laneq_u16(v24_31, vb1, va0, 3);     \
      v32_39 = vmlaq_laneq_u16(v32_39, vb1, va0, 4);     \
      v40_47 = vmlaq_laneq_u16(v40_47, vb1, va0, 5);     \
      v48_55 = vmlaq_laneq_u16(v48_55, vb1, va0, 6);     \
      v56_63 = vmlaq_laneq_u16(v56_63, vb1, va0, 7);     \
      v64_71 = vmlaq_laneq_u16(v64_71, vb2, va0, 0);     \
      v72_79 = vmlaq_laneq_u16(v72_79, vb2, va0, 1);     \
      v80_87 = vmlaq_laneq_u16(v80_87, vb2, va0, 2);     \
      v88_95 = vmlaq_laneq_u16(v88_95, vb2, va0, 3);     \
      v96_103 = vmlaq_laneq_u16(v96_103, vb2, va0, 4);   \
      v104_111 = vmlaq_laneq_u16(v104_111, vb2, va0, 5); \
      v112_119 = vmlaq_laneq_u16(v112_119, vb2, va0, 6); \
      v120_127 = vmlaq_laneq_u16(v120_127, vb2, va0, 7); \
      va0 = vld1q_u16(a + 8 * i + 8 * 2);                \
      vb1 = vld1q_u16(b + 16 * i + 8 * 4);               \
      vb2 = vld1q_u16(b + 16 * i + 8 * 5);               \
      v0_7 = vmlaq_laneq_u16(v0_7, vb1, va0, 0);         \
      v8_15 = vmlaq_laneq_u16(v8_15, vb1, va0, 1);       \
      v16_23 = vmlaq_laneq_u16(v16_23, vb1, va0, 2);     \
      v24_31 = vmlaq_laneq_u16(v24_31, vb1, va0, 3);     \
      v32_39 = vmlaq_laneq_u16(v32_39, vb1, va0, 4);     \
      v40_47 = vmlaq_laneq_u16(v40_47, vb1, va0, 5);     \
      v48_55 = vmlaq_laneq_u16(v48_55, vb1, va0, 6);     \
      v56_63 = vmlaq_laneq_u16(v56_63, vb1, va0, 7);     \
      v64_71 = vmlaq_laneq_u16(v64_71, vb2, va0, 0);     \
      v72_79 = vmlaq_laneq_u16(v72_79, vb2, va0, 1);     \
      v80_87 = vmlaq_laneq_u16(v80_87, vb2, va0, 2);     \
      v88_95 = vmlaq_laneq_u16(v88_95, vb2, va0, 3);     \
      v96_103 = vmlaq_laneq_u16(v96_103, vb2, va0, 4);   \
      v104_111 = vmlaq_laneq_u16(v104_111, vb2, va0, 5); \
      v112_119 = vmlaq_laneq_u16(v112_119, vb2, va0, 6); \
      v120_127 = vmlaq_laneq_u16(v120_127, vb2, va0, 7); \
      va0 = vld1q_u16(a + 8 * i + 8 * 3);                \
      vb1 = vld1q_u16(b + 16 * i + 8 * 6);               \
      vb2 = vld1q_u16(b + 16 * i + 8 * 7);               \
      v0_7 = vmlaq_laneq_u16(v0_7, vb1, va0, 0);         \
      v8_15 = vmlaq_laneq_u16(v8_15, vb1, va0, 1);       \
      v16_23 = vmlaq_laneq_u16(v16_23, vb1, va0, 2);     \
      v24_31 = vmlaq_laneq_u16(v24_31, vb1, va0, 3);     \
      v32_39 = vmlaq_laneq_u16(v32_39, vb1, va0, 4);     \
      v40_47 = vmlaq_laneq_u16(v40_47, vb1, va0, 5);     \
      v48_55 = vmlaq_laneq_u16(v48_55, vb1, va0, 6);     \
      v56_63 = vmlaq_laneq_u16(v56_63, vb1, va0, 7);     \
      v64_71 = vmlaq_laneq_u16(v64_71, vb2, va0, 0);     \
      v72_79 = vmlaq_laneq_u16(v72_79, vb2, va0, 1);     \
      v80_87 = vmlaq_laneq_u16(v80_87, vb2, va0, 2);     \
      v88_95 = vmlaq_laneq_u16(v88_95, vb2, va0, 3);     \
      v96_103 = vmlaq_laneq_u16(v96_103, vb2, va0, 4);   \
      v104_111 = vmlaq_laneq_u16(v104_111, vb2, va0, 5); \
      v112_119 = vmlaq_laneq_u16(v112_119, vb2, va0, 6); \
      v120_127 = vmlaq_laneq_u16(v120_127, vb2, va0, 7); \
    }                                                    \
    l += N;                                              \
    __builtin_prefetch(b + 16 * N, 0, 3);                \
    __builtin_prefetch(a + 8 * N, 0, 3);                 \
    b += 16 * N;                                         \
    a += 8 * N;                                          \
  } while (0)

#define KERNEL_8x16_ACC1()                             \
  do {                                                 \
    va0 = vld1q_u16(a);                                \
    vb1 = vld1q_u16(b);                                \
    vb2 = vld1q_u16(b + 8);                            \
    v0_7 = vmlaq_laneq_u16(v0_7, vb1, va0, 0);         \
    v8_15 = vmlaq_laneq_u16(v8_15, vb1, va0, 1);       \
    v16_23 = vmlaq_laneq_u16(v16_23, vb1, va0, 2);     \
    v24_31 = vmlaq_laneq_u16(v24_31, vb1, va0, 3);     \
    v32_39 = vmlaq_laneq_u16(v32_39, vb1, va0, 4);     \
    v40_47 = vmlaq_laneq_u16(v40_47, vb1, va0, 5);     \
    v48_55 = vmlaq_laneq_u16(v48_55, vb1, va0, 6);     \
    v56_63 = vmlaq_laneq_u16(v56_63, vb1, va0, 7);     \
    v64_71 = vmlaq_laneq_u16(v64_71, vb2, va0, 0);     \
    v72_79 = vmlaq_laneq_u16(v72_79, vb2, va0, 1);     \
    v80_87 = vmlaq_laneq_u16(v80_87, vb2, va0, 2);     \
    v88_95 = vmlaq_laneq_u16(v88_95, vb2, va0, 3);     \
    v96_103 = vmlaq_laneq_u16(v96_103, vb2, va0, 4);   \
    v104_111 = vmlaq_laneq_u16(v104_111, vb2, va0, 5); \
    v112_119 = vmlaq_laneq_u16(v112_119, vb2, va0, 6); \
    v120_127 = vmlaq_laneq_u16(v120_127, vb2, va0, 7); \
    l += 1;                                            \
    __builtin_prefetch(b + 16, 0, 3);                  \
    __builtin_prefetch(a + 8, 0, 3);                   \
    b += 16 * 1;                                       \
    a += 8 * 1;                                        \
  } while (0)

#define SAVE_KERNEL_8X16_u16_u32()                                             \
  do {                                                                         \
    vst1q_u32(c, vaddq_u32(vld1q_u32(c), vmovl_u16(vget_low_u16(v0_7))));   \
    vst1q_u32(c + 4,                                                           \
              vaddq_u32(vld1q_u32(c + 4), vmovl_u16(vget_high_u16(v0_7)))); \
                                                                               \
    vst1q_u32(                                                                 \
      c + 8, vaddq_u32(vld1q_u32(c + 8), vmovl_u16(vget_low_u16(v64_71)))); \
    vst1q_u32(c + 8 + 4, vaddq_u32(vld1q_u32(c + 8 + 4),                       \
                                   vmovl_u16(vget_high_u16(v64_71))));      \
                                                                               \
    vst1q_u32(c + ldc, vaddq_u32(vld1q_u32(c + ldc),                           \
                                 vmovl_u16(vget_low_u16(v8_15))));          \
    vst1q_u32(c + ldc + 4, vaddq_u32(vld1q_u32(c + ldc + 4),                   \
                                     vmovl_u16(vget_high_u16(v8_15))));     \
                                                                               \
    vst1q_u32(c + ldc + 8, vaddq_u32(vld1q_u32(c + ldc + 8),                   \
                                     vmovl_u16(vget_low_u16(v72_79))));     \
    vst1q_u32(c + ldc + 8 + 4,                                                 \
              vaddq_u32(vld1q_u32(c + ldc + 8 + 4),                            \
                        vmovl_u16(vget_high_u16(v72_79))));                 \
                                                                               \
    vst1q_u32(c + 2 * ldc, vaddq_u32(vld1q_u32(c + 2 * ldc),                   \
                                     vmovl_u16(vget_low_u16(v16_23))));     \
    vst1q_u32(c + 2 * ldc + 4,                                                 \
              vaddq_u32(vld1q_u32(c + 2 * ldc + 4),                            \
                        vmovl_u16(vget_high_u16(v16_23))));                 \
                                                                               \
    vst1q_u32(c + 2 * ldc + 8, vaddq_u32(vld1q_u32(c + 2 * ldc + 8),           \
                                         vmovl_u16(vget_low_u16(v80_87)))); \
    vst1q_u32(c + 2 * ldc + 8 + 4,                                             \
              vaddq_u32(vld1q_u32(c + 2 * ldc + 8 + 4),                        \
                        vmovl_u16(vget_high_u16(v80_87))));                 \
                                                                               \
    vst1q_u32(c + 3 * ldc, vaddq_u32(vld1q_u32(c + 3 * ldc),                   \
                                     vmovl_u16(vget_low_u16(v24_31))));     \
    vst1q_u32(c + 3 * ldc + 4,                                                 \
              vaddq_u32(vld1q_u32(c + 3 * ldc + 4),                            \
                        vmovl_u16(vget_high_u16(v24_31))));                 \
                                                                               \
    vst1q_u32(c + 3 * ldc + 8, vaddq_u32(vld1q_u32(c + 3 * ldc + 8),           \
                                         vmovl_u16(vget_low_u16(v88_95)))); \
    vst1q_u32(c + 3 * ldc + 8 + 4,                                             \
              vaddq_u32(vld1q_u32(c + 3 * ldc + 8 + 4),                        \
                        vmovl_u16(vget_high_u16(v88_95))));                 \
                                                                               \
    vst1q_u32(c + 4 * ldc, vaddq_u32(vld1q_u32(c + 4 * ldc),                   \
                                     vmovl_u16(vget_low_u16(v32_39))));     \
    vst1q_u32(c + 4 * ldc + 4,                                                 \
              vaddq_u32(vld1q_u32(c + 4 * ldc + 4),                            \
                        vmovl_u16(vget_high_u16(v32_39))));                 \
                                                                               \
    vst1q_u32(c + 4 * ldc + 8,                                                 \
              vaddq_u32(vld1q_u32(c + 4 * ldc + 8),                            \
                        vmovl_u16(vget_low_u16(v96_103))));                 \
    vst1q_u32(c + 4 * ldc + 8 + 4,                                             \
              vaddq_u32(vld1q_u32(c + 4 * ldc + 8 + 4),                        \
                        vmovl_u16(vget_high_u16(v96_103))));                \
                                                                               \
    vst1q_u32(c + 5 * ldc, vaddq_u32(vld1q_u32(c + 5 * ldc),                   \
                                     vmovl_u16(vget_low_u16(v40_47))));     \
    vst1q_u32(c + 5 * ldc + 4,                                                 \
              vaddq_u32(vld1q_u32(c + 5 * ldc + 4),                            \
                        vmovl_u16(vget_high_u16(v40_47))));                 \
    vst1q_u32(c + 5 * ldc + 8,                                                 \
              vaddq_u32(vld1q_u32(c + 5 * ldc + 8),                            \
                        vmovl_u16(vget_low_u16(v104_111))));                \
    vst1q_u32(c + 5 * ldc + 8 + 4,                                             \
              vaddq_u32(vld1q_u32(c + 5 * ldc + 8 + 4),                        \
                        vmovl_u16(vget_high_u16(v104_111))));               \
                                                                               \
    vst1q_u32(c + 6 * ldc, vaddq_u32(vld1q_u32(c + 6 * ldc),                   \
                                     vmovl_u16(vget_low_u16(v48_55))));     \
    vst1q_u32(c + 6 * ldc + 4,                                                 \
              vaddq_u32(vld1q_u32(c + 6 * ldc + 4),                            \
                        vmovl_u16(vget_high_u16(v48_55))));                 \
                                                                               \
    vst1q_u32(c + 6 * ldc + 8,                                                 \
              vaddq_u32(vld1q_u32(c + 6 * ldc + 8),                            \
                        vmovl_u16(vget_low_u16(v112_119))));                \
    vst1q_u32(c + 6 * ldc + 8 + 4,                                             \
              vaddq_u32(vld1q_u32(c + 6 * ldc + 8 + 4),                        \
                        vmovl_u16(vget_high_u16(v112_119))));               \
                                                                               \
    vst1q_u32(c + 7 * ldc, vaddq_u32(vld1q_u32(c + 7 * ldc),                   \
                                     vmovl_u16(vget_low_u16(v56_63))));     \
    vst1q_u32(c + 7 * ldc + 4,                                                 \
              vaddq_u32(vld1q_u32(c + 7 * ldc + 4),                            \
                        vmovl_u16(vget_high_u16(v56_63))));                 \
                                                                               \
    vst1q_u32(c + 7 * ldc + 8,                                                 \
              vaddq_u32(vld1q_u32(c + 7 * ldc + 8),                            \
                        vmovl_u16(vget_low_u16(v120_127))));                \
    vst1q_u32(c + 7 * ldc + 8 + 4,                                             \
              vaddq_u32(vld1q_u32(c + 7 * ldc + 8 + 4),                        \
                        vmovl_u16(vget_high_u16(v120_127))));               \
  } while (0)

template <>
void uhgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
                       uint16_t *sa, uint16_t *sb, uint16_t *sc, unsigned int ldc) {
  throw std::runtime_error(
    "Error : should not reach for full-fp16 usage in experimental kernel");
}

template <>
void uhgemm_kernel_8x16(unsigned int M, unsigned int N, unsigned int K,
                       uint16_t *sa, uint16_t *sb, unsigned int *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 16 == 0 && K % 4 == 0);

  uint16_t *a = sa, *b = sb;
  unsigned int *c = sc;
  unsigned int i, j, l;
  unsigned int K4 = get_prev_mltpl_of_2p_n(K, 2);
  unsigned int K8 = get_prev_mltpl_of_2p_n(K, 3);
  unsigned int K16 = get_prev_mltpl_of_2p_n(K, 4);
  unsigned int K32 = get_prev_mltpl_of_2p_n(K, 5);
  unsigned int K64 = get_prev_mltpl_of_2p_n(K, 6);

  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 16) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      uint16x8_t v0_7, v8_15;
      uint16x8_t v16_23, v24_31;
      uint16x8_t v32_39, v40_47;
      uint16x8_t v48_55, v56_63;
      uint16x8_t v64_71, v72_79;
      uint16x8_t v80_87, v88_95;
      uint16x8_t v96_103, v104_111;
      uint16x8_t v112_119, v120_127;
      uint16x8_t va0;
      uint16x8_t vb1, vb2;
      l = 0;
      for (; l < K64;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC_N4(64);
        SAVE_KERNEL_8X16_u16_u32();
      }
      for (; l < K32;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC_N4(32);
        SAVE_KERNEL_8X16_u16_u32();
      }
      for (; l < K16;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC_N4(16);
        SAVE_KERNEL_8X16_u16_u32();
      }
      for (; l < K8;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC_N4(8);
        SAVE_KERNEL_8X16_u16_u32();
      }
      for (; l < K4;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC_N4(4);
        SAVE_KERNEL_8X16_u16_u32();
      }
      for (; l < K;) {
        INIT_KERNEL_8X16();
        KERNEL_8x16_ACC1();
        SAVE_KERNEL_8X16_u16_u32();
      }
      c += 16;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}
