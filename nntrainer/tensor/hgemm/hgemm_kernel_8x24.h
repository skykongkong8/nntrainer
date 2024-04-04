// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_kernel_8x24.h
 * @date   04 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM 8x16 kernel
 *
 */

#include <hgemm_common.h>
#include <stdlib.h>

/// @note Following KERNELs are the combinations of accuracy-latency
/// tradeoff. User can select which kernel to use by replacing them.

// 1. Partial sum 1536 digits : Worst accuracy, best latency
#define KERNEL_8x24_ACC8()                           \
  v0_7 = vdupq_n_f16(0.F);                           \
  v8_15 = vdupq_n_f16(0.F);                          \
  v16_23 = vdupq_n_f16(0.F);                         \
  v24_31 = vdupq_n_f16(0.F);                         \
  v32_39 = vdupq_n_f16(0.F);                         \
  v40_47 = vdupq_n_f16(0.F);                         \
  v48_55 = vdupq_n_f16(0.F);                         \
  v56_63 = vdupq_n_f16(0.F);                         \
  v64_71 = vdupq_n_f16(0.F);                         \
  v72_79 = vdupq_n_f16(0.F);                         \
  v80_87 = vdupq_n_f16(0.F);                         \
  v88_95 = vdupq_n_f16(0.F);                         \
  v96_103 = vdupq_n_f16(0.F);                        \
  v104_111 = vdupq_n_f16(0.F);                       \
  v112_119 = vdupq_n_f16(0.F);                       \
  v120_127 = vdupq_n_f16(0.F);                       \
  v128_135 = vdupq_n_f16(0.F);                       \
  v136_143 = vdupq_n_f16(0.F);                       \
  v144_151 = vdupq_n_f16(0.F);                       \
  v152_159 = vdupq_n_f16(0.F);                       \
  v160_167 = vdupq_n_f16(0.F);                       \
  v168_175 = vdupq_n_f16(0.F);                       \
  v176_183 = vdupq_n_f16(0.F);                       \
  v184_191 = vdupq_n_f16(0.F);                       \
  va0 = vld1q_f16(a);                                \
  v24 = vld1q_f16(b);                                \
  v25 = vld1q_f16(b + 8);                            \
  v26 = vld1q_f16(b + 16);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 8);                            \
  v24 = vld1q_f16(b + 24);                           \
  v25 = vld1q_f16(b + 32);                           \
  v26 = vld1q_f16(b + 40);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 16);                           \
  v24 = vld1q_f16(b + 48);                           \
  v25 = vld1q_f16(b + 56);                           \
  v26 = vld1q_f16(b + 64);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 24);                           \
  v24 = vld1q_f16(b + 72);                           \
  v25 = vld1q_f16(b + 80);                           \
  v26 = vld1q_f16(b + 88);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 32);                           \
  v24 = vld1q_f16(b + 96);                           \
  v25 = vld1q_f16(b + 104);                          \
  v26 = vld1q_f16(b + 112);                          \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 40);                           \
  v24 = vld1q_f16(b + 120);                          \
  v25 = vld1q_f16(b + 128);                          \
  v26 = vld1q_f16(b + 136);                          \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 48);                           \
  v24 = vld1q_f16(b + 144);                          \
  v25 = vld1q_f16(b + 152);                          \
  v26 = vld1q_f16(b + 160);                          \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 56);                           \
  v24 = vld1q_f16(b + 168);                          \
  v25 = vld1q_f16(b + 176);                          \
  v26 = vld1q_f16(b + 184);                          \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  l += 8;                                            \
  __builtin_prefetch(b + 192, 0, 3);                 \
  __builtin_prefetch(a + 64, 0, 3);                  \
  b += 24 * 8;                                       \
  a += 8 * 8;

// 2. Partial sum 768 digits : Medium accuracy, medium latency
#define KERNEL_8x24_ACC4()                           \
  v0_7 = vdupq_n_f16(0.F);                           \
  v8_15 = vdupq_n_f16(0.F);                          \
  v16_23 = vdupq_n_f16(0.F);                         \
  v24_31 = vdupq_n_f16(0.F);                         \
  v32_39 = vdupq_n_f16(0.F);                         \
  v40_47 = vdupq_n_f16(0.F);                         \
  v48_55 = vdupq_n_f16(0.F);                         \
  v56_63 = vdupq_n_f16(0.F);                         \
  v64_71 = vdupq_n_f16(0.F);                         \
  v72_79 = vdupq_n_f16(0.F);                         \
  v80_87 = vdupq_n_f16(0.F);                         \
  v88_95 = vdupq_n_f16(0.F);                         \
  v96_103 = vdupq_n_f16(0.F);                        \
  v104_111 = vdupq_n_f16(0.F);                       \
  v112_119 = vdupq_n_f16(0.F);                       \
  v120_127 = vdupq_n_f16(0.F);                       \
  v128_135 = vdupq_n_f16(0.F);                       \
  v136_143 = vdupq_n_f16(0.F);                       \
  v144_151 = vdupq_n_f16(0.F);                       \
  v152_159 = vdupq_n_f16(0.F);                       \
  v160_167 = vdupq_n_f16(0.F);                       \
  v168_175 = vdupq_n_f16(0.F);                       \
  v176_183 = vdupq_n_f16(0.F);                       \
  v184_191 = vdupq_n_f16(0.F);                       \
  va0 = vld1q_f16(a);                                \
  v24 = vld1q_f16(b);                                \
  v25 = vld1q_f16(b + 8);                            \
  v26 = vld1q_f16(b + 16);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 8);                            \
  v24 = vld1q_f16(b + 24);                           \
  v25 = vld1q_f16(b + 32);                           \
  v26 = vld1q_f16(b + 40);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 16);                           \
  v24 = vld1q_f16(b + 48);                           \
  v25 = vld1q_f16(b + 56);                           \
  v26 = vld1q_f16(b + 64);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  va0 = vld1q_f16(a + 24);                           \
  v24 = vld1q_f16(b + 72);                           \
  v25 = vld1q_f16(b + 80);                           \
  v26 = vld1q_f16(b + 88);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  l += 4;                                            \
  __builtin_prefetch(b + 96, 0, 3);                  \
  __builtin_prefetch(a + 32, 0, 3);                  \
  b += 24 * 4;                                       \
  a += 8 * 4;

// 3. Partial sum 192 digits : Best accuracy, worst latency
#define KERNEL_8x24_ACC1()                           \
  v0_7 = vdupq_n_f16(0.F);                           \
  v8_15 = vdupq_n_f16(0.F);                          \
  v16_23 = vdupq_n_f16(0.F);                         \
  v24_31 = vdupq_n_f16(0.F);                         \
  v32_39 = vdupq_n_f16(0.F);                         \
  v40_47 = vdupq_n_f16(0.F);                         \
  v48_55 = vdupq_n_f16(0.F);                         \
  v56_63 = vdupq_n_f16(0.F);                         \
  v64_71 = vdupq_n_f16(0.F);                         \
  v72_79 = vdupq_n_f16(0.F);                         \
  v80_87 = vdupq_n_f16(0.F);                         \
  v88_95 = vdupq_n_f16(0.F);                         \
  v96_103 = vdupq_n_f16(0.F);                        \
  v104_111 = vdupq_n_f16(0.F);                       \
  v112_119 = vdupq_n_f16(0.F);                       \
  v120_127 = vdupq_n_f16(0.F);                       \
  v128_135 = vdupq_n_f16(0.F);                       \
  v136_143 = vdupq_n_f16(0.F);                       \
  v144_151 = vdupq_n_f16(0.F);                       \
  v152_159 = vdupq_n_f16(0.F);                       \
  v160_167 = vdupq_n_f16(0.F);                       \
  v168_175 = vdupq_n_f16(0.F);                       \
  v176_183 = vdupq_n_f16(0.F);                       \
  v184_191 = vdupq_n_f16(0.F);                       \
  va0 = vld1q_f16(a);                                \
  v24 = vld1q_f16(b);                                \
  v25 = vld1q_f16(b + 8);                            \
  v26 = vld1q_f16(b + 16);                           \
  v0_7 = vfmaq_laneq_f16(v0_7, v24, va0, 0);         \
  v8_15 = vfmaq_laneq_f16(v8_15, v24, va0, 1);       \
  v16_23 = vfmaq_laneq_f16(v16_23, v24, va0, 2);     \
  v24_31 = vfmaq_laneq_f16(v24_31, v24, va0, 3);     \
  v32_39 = vfmaq_laneq_f16(v32_39, v24, va0, 4);     \
  v40_47 = vfmaq_laneq_f16(v40_47, v24, va0, 5);     \
  v48_55 = vfmaq_laneq_f16(v48_55, v24, va0, 6);     \
  v56_63 = vfmaq_laneq_f16(v56_63, v24, va0, 7);     \
  v64_71 = vfmaq_laneq_f16(v64_71, v25, va0, 0);     \
  v72_79 = vfmaq_laneq_f16(v72_79, v25, va0, 1);     \
  v80_87 = vfmaq_laneq_f16(v80_87, v25, va0, 2);     \
  v88_95 = vfmaq_laneq_f16(v88_95, v25, va0, 3);     \
  v96_103 = vfmaq_laneq_f16(v96_103, v25, va0, 4);   \
  v104_111 = vfmaq_laneq_f16(v104_111, v25, va0, 5); \
  v112_119 = vfmaq_laneq_f16(v112_119, v25, va0, 6); \
  v120_127 = vfmaq_laneq_f16(v120_127, v25, va0, 7); \
  v128_135 = vfmaq_laneq_f16(v128_135, v26, va0, 0); \
  v136_143 = vfmaq_laneq_f16(v136_143, v26, va0, 1); \
  v144_151 = vfmaq_laneq_f16(v144_151, v26, va0, 2); \
  v152_159 = vfmaq_laneq_f16(v152_159, v26, va0, 3); \
  v160_167 = vfmaq_laneq_f16(v160_167, v26, va0, 4); \
  v168_175 = vfmaq_laneq_f16(v168_175, v26, va0, 5); \
  v176_183 = vfmaq_laneq_f16(v176_183, v26, va0, 6); \
  v184_191 = vfmaq_laneq_f16(v184_191, v26, va0, 7); \
  l += 1;                                            \
  __builtin_prefetch(b + 24, 0, 3);                  \
  __builtin_prefetch(a + 8, 0, 3);                   \
  b += 24 * 1;                                       \
  a += 8 * 1;

/**
 * @brief hgemm 8x16 kernel sc = sa * sb
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_8x24(unsigned int M, unsigned int N, unsigned int K,
                       __fp16 *sa, __fp16 *sb, __fp16 *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 24 == 0);

  __fp16 *a = sa, *b = sb, *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 24) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      // 8x24
      float16x8_t v0_7, v8_15, v16_23;
      float16x8_t v24_31, v32_39, v40_47;
      float16x8_t v48_55, v56_63, v64_71;
      float16x8_t v72_79, v80_87, v88_95;
      float16x8_t v96_103, v104_111, v112_119;
      float16x8_t v120_127, v128_135, v136_143;
      float16x8_t v144_151, v152_159, v160_167;
      float16x8_t v168_175, v176_183, v184_191;

      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x8_t va0, va1, va2, va3;
      l = 0;
      for (; l < K;) {
        KERNEL_8x24_ACC8();
        vst1q_f16(c, vaddq_f16(vld1q_f16(c), v0_7));
        vst1q_f16(c + 8, vaddq_f16(vld1q_f16(c + 8), v64_71));
        vst1q_f16(c + 16, vaddq_f16(vld1q_f16(c + 16), v128_135));

        vst1q_f16(c + ldc, vaddq_f16(vld1q_f16(c + ldc), v8_15));
        vst1q_f16(c + ldc + 8, vaddq_f16(vld1q_f16(c + ldc + 8), v72_79));
        vst1q_f16(c + ldc + 16, vaddq_f16(vld1q_f16(c + ldc + 16), v136_143));

        vst1q_f16(c + 2 * ldc, vaddq_f16(vld1q_f16(c + 2 * ldc), v16_23));
        vst1q_f16(c + 2 * ldc + 8,
                  vaddq_f16(vld1q_f16(c + 2 * ldc + 8), v80_87));
        vst1q_f16(c + 2 * ldc + 16,
                  vaddq_f16(vld1q_f16(c + 2 * ldc + 16), v144_151));

        vst1q_f16(c + 3 * ldc, vaddq_f16(vld1q_f16(c + 3 * ldc), v24_31));
        vst1q_f16(c + 3 * ldc + 8,
                  vaddq_f16(vld1q_f16(c + 3 * ldc + 8), v88_95));
        vst1q_f16(c + 3 * ldc + 16,
                  vaddq_f16(vld1q_f16(c + 3 * ldc + 16), v152_159));

        vst1q_f16(c + 4 * ldc, vaddq_f16(vld1q_f16(c + 4 * ldc), v32_39));
        vst1q_f16(c + 4 * ldc + 8,
                  vaddq_f16(vld1q_f16(c + 4 * ldc + 8), v96_103));
        vst1q_f16(c + 4 * ldc + 16,
                  vaddq_f16(vld1q_f16(c + 4 * ldc + 16), v160_167));

        vst1q_f16(c + 5 * ldc, vaddq_f16(vld1q_f16(c + 5 * ldc), v40_47));
        vst1q_f16(c + 5 * ldc + 8,
                  vaddq_f16(vld1q_f16(c + 5 * ldc + 8), v104_111));
        vst1q_f16(c + 5 * ldc + 16,
                  vaddq_f16(vld1q_f16(c + 5 * ldc + 16), v168_175));

        vst1q_f16(c + 6 * ldc, vaddq_f16(vld1q_f16(c + 6 * ldc), v48_55));
        vst1q_f16(c + 6 * ldc + 8,
                  vaddq_f16(vld1q_f16(c + 6 * ldc + 8), v112_119));
        vst1q_f16(c + 6 * ldc + 16,
                  vaddq_f16(vld1q_f16(c + 6 * ldc + 16), v176_183));

        vst1q_f16(c + 7 * ldc, vaddq_f16(vld1q_f16(c + 7 * ldc), v56_63));
        vst1q_f16(c + 7 * ldc + 8,
                  vaddq_f16(vld1q_f16(c + 7 * ldc + 8), v120_127));
        vst1q_f16(c + 7 * ldc + 16,
                  vaddq_f16(vld1q_f16(c + 7 * ldc + 16), v184_191));
      }
      c += 24;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}

/**
 * @brief hgemm 8x16 kernel sc = sa * sb
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param sa sub-matrix of input matrix A
 * @param sb sub-matrix of input matrix B
 * @param sc sub-matrix of output matrix C
 * @param ldc leading-dimension of matrix C
 */
void hgemm_kernel_8x24(unsigned int M, unsigned int N, unsigned int K,
                       __fp16 *sa, __fp16 *sb, float *sc, unsigned int ldc) {
  assert(M > 0 && N > 0 && K > 0);
  assert(M % 8 == 0 && N % 24 == 0);

  __fp16 *a = sa, *b = sb;
  float *c = sc;
  unsigned int i, j, l;
  for (i = 0; i < M; i += 8) {
    for (j = 0; j < N; j += 24) {
      __builtin_prefetch(b, 0, 3);
      __builtin_prefetch(a, 0, 3);
      // 3x24
      float16x8_t v0_7, v8_15, v16_23;
      float16x8_t v24_31, v32_39, v40_47;
      float16x8_t v48_55, v56_63, v64_71;
      float16x8_t v72_79, v80_87, v88_95;
      float16x8_t v96_103, v104_111, v112_119;
      float16x8_t v120_127, v128_135, v136_143;
      float16x8_t v144_151, v152_159, v160_167;
      float16x8_t v168_175, v176_183, v184_191;

      float16x8_t v24, v25, v26, v27, v28, v29, v30, v31;
      float16x8_t va0, va1, va2, va3;
      l = 0;
      for (; l < K;) {
        KERNEL_8x24_ACC8();

        vst1q_f32(c, vaddq_f32(vld1q_f32(c), vcvt_f32_f16(vget_low_f16(v0_7))));
        vst1q_f32(c + 4, vaddq_f32(vld1q_f32(c + 4),
                                   vcvt_f32_f16(vget_high_f16(v0_7))));

        vst1q_f32(c + 8, vaddq_f32(vld1q_f32(c + 8),
                                   vcvt_f32_f16(vget_low_f16(v64_71))));
        vst1q_f32(c + 8 + 4, vaddq_f32(vld1q_f32(c + 8 + 4),
                                       vcvt_f32_f16(vget_high_f16(v64_71))));

        vst1q_f32(c + 16, vaddq_f32(vld1q_f32(c + 16),
                                    vcvt_f32_f16(vget_low_f16(v128_135))));
        vst1q_f32(c + 16 + 4, vaddq_f32(vld1q_f32(c + 16 + 4),
                                        vcvt_f32_f16(vget_high_f16(v128_135))));

        vst1q_f32(c + ldc, vaddq_f32(vld1q_f32(c + ldc),
                                     vcvt_f32_f16(vget_low_f16(v8_15))));
        vst1q_f32(c + ldc + 4, vaddq_f32(vld1q_f32(c + ldc + 4),
                                         vcvt_f32_f16(vget_high_f16(v8_15))));

        vst1q_f32(c + ldc + 8, vaddq_f32(vld1q_f32(c + ldc + 8),
                                         vcvt_f32_f16(vget_low_f16(v72_79))));
        vst1q_f32(c + ldc + 8 + 4,
                  vaddq_f32(vld1q_f32(c + ldc + 8 + 4),
                            vcvt_f32_f16(vget_high_f16(v72_79))));

        vst1q_f32(c + ldc + 16,
                  vaddq_f32(vld1q_f32(c + ldc + 16),
                            vcvt_f32_f16(vget_low_f16(v136_143))));
        vst1q_f32(c + ldc + 16 + 4,
                  vaddq_f32(vld1q_f32(c + ldc + 16 + 4),
                            vcvt_f32_f16(vget_high_f16(v136_143))));

        vst1q_f32(c + 2 * ldc, vaddq_f32(vld1q_f32(c + 2 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v16_23))));
        vst1q_f32(c + 2 * ldc + 4,
                  vaddq_f32(vld1q_f32(c + 2 * ldc + 4),
                            vcvt_f32_f16(vget_high_f16(v16_23))));

        vst1q_f32(c + 2 * ldc + 8,
                  vaddq_f32(vld1q_f32(c + 2 * ldc + 8),
                            vcvt_f32_f16(vget_low_f16(v80_87))));
        vst1q_f32(c + 2 * ldc + 8 + 4,
                  vaddq_f32(vld1q_f32(c + 2 * ldc + 8 + 4),
                            vcvt_f32_f16(vget_high_f16(v80_87))));

        vst1q_f32(c + 2 * ldc + 16,
                  vaddq_f32(vld1q_f32(c + 2 * ldc + 16),
                            vcvt_f32_f16(vget_low_f16(v144_151))));
        vst1q_f32(c + 2 * ldc + 16 + 4,
                  vaddq_f32(vld1q_f32(c + 2 * ldc + 16 + 4),
                            vcvt_f32_f16(vget_high_f16(v144_151))));

        vst1q_f32(c + 3 * ldc, vaddq_f32(vld1q_f32(c + 3 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v24_31))));
        vst1q_f32(c + 3 * ldc + 4,
                  vaddq_f32(vld1q_f32(c + 3 * ldc + 4),
                            vcvt_f32_f16(vget_high_f16(v24_31))));

        vst1q_f32(c + 3 * ldc + 8,
                  vaddq_f32(vld1q_f32(c + 3 * ldc + 8),
                            vcvt_f32_f16(vget_low_f16(v88_95))));
        vst1q_f32(c + 3 * ldc + 8 + 4,
                  vaddq_f32(vld1q_f32(c + 3 * ldc + 8 + 4),
                            vcvt_f32_f16(vget_high_f16(v88_95))));

        vst1q_f32(c + 3 * ldc + 16,
                  vaddq_f32(vld1q_f32(c + 3 * ldc + 16),
                            vcvt_f32_f16(vget_low_f16(v152_159))));
        vst1q_f32(c + 3 * ldc + 16 + 4,
                  vaddq_f32(vld1q_f32(c + 3 * ldc + 16 + 4),
                            vcvt_f32_f16(vget_high_f16(v152_159))));

        vst1q_f32(c + 4 * ldc, vaddq_f32(vld1q_f32(c + 4 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v32_39))));
        vst1q_f32(c + 4 * ldc + 4,
                  vaddq_f32(vld1q_f32(c + 4 * ldc + 4),
                            vcvt_f32_f16(vget_high_f16(v32_39))));

        vst1q_f32(c + 4 * ldc + 8,
                  vaddq_f32(vld1q_f32(c + 4 * ldc + 8),
                            vcvt_f32_f16(vget_low_f16(v96_103))));
        vst1q_f32(c + 4 * ldc + 8 + 4,
                  vaddq_f32(vld1q_f32(c + 4 * ldc + 8 + 4),
                            vcvt_f32_f16(vget_high_f16(v96_103))));

        vst1q_f32(c + 4 * ldc + 16,
                  vaddq_f32(vld1q_f32(c + 4 * ldc + 16),
                            vcvt_f32_f16(vget_low_f16(v160_167))));
        vst1q_f32(c + 4 * ldc + 16 + 4,
                  vaddq_f32(vld1q_f32(c + 4 * ldc + 16 + 4),
                            vcvt_f32_f16(vget_high_f16(v160_167))));

        vst1q_f32(c + 5 * ldc, vaddq_f32(vld1q_f32(c + 5 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v40_47))));
        vst1q_f32(c + 5 * ldc + 4,
                  vaddq_f32(vld1q_f32(c + 5 * ldc + 4),
                            vcvt_f32_f16(vget_high_f16(v40_47))));

        vst1q_f32(c + 5 * ldc + 8,
                  vaddq_f32(vld1q_f32(c + 5 * ldc + 8),
                            vcvt_f32_f16(vget_low_f16(v104_111))));
        vst1q_f32(c + 5 * ldc + 8 + 4,
                  vaddq_f32(vld1q_f32(c + 5 * ldc + 8 + 4),
                            vcvt_f32_f16(vget_high_f16(v104_111))));

        vst1q_f32(c + 5 * ldc + 16,
                  vaddq_f32(vld1q_f32(c + 5 * ldc + 16),
                            vcvt_f32_f16(vget_low_f16(v168_175))));
        vst1q_f32(c + 5 * ldc + 16 + 4,
                  vaddq_f32(vld1q_f32(c + 5 * ldc + 16 + 4),
                            vcvt_f32_f16(vget_high_f16(v168_175))));

        vst1q_f32(c + 6 * ldc, vaddq_f32(vld1q_f32(c + 6 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v48_55))));
        vst1q_f32(c + 6 * ldc + 4,
                  vaddq_f32(vld1q_f32(c + 6 * ldc + 4),
                            vcvt_f32_f16(vget_high_f16(v48_55))));

        vst1q_f32(c + 6 * ldc + 8,
                  vaddq_f32(vld1q_f32(c + 6 * ldc + 8),
                            vcvt_f32_f16(vget_low_f16(v112_119))));
        vst1q_f32(c + 6 * ldc + 8 + 4,
                  vaddq_f32(vld1q_f32(c + 6 * ldc + 8 + 4),
                            vcvt_f32_f16(vget_high_f16(v112_119))));

        vst1q_f32(c + 6 * ldc + 16,
                  vaddq_f32(vld1q_f32(c + 6 * ldc + 16),
                            vcvt_f32_f16(vget_low_f16(v176_183))));
        vst1q_f32(c + 6 * ldc + 16 + 4,
                  vaddq_f32(vld1q_f32(c + 6 * ldc + 16 + 4),
                            vcvt_f32_f16(vget_high_f16(v176_183))));

        vst1q_f32(c + 7 * ldc, vaddq_f32(vld1q_f32(c + 7 * ldc),
                                         vcvt_f32_f16(vget_low_f16(v56_63))));
        vst1q_f32(c + 7 * ldc + 4,
                  vaddq_f32(vld1q_f32(c + 7 * ldc + 4),
                            vcvt_f32_f16(vget_high_f16(v56_63))));

        vst1q_f32(c + 7 * ldc + 8,
                  vaddq_f32(vld1q_f32(c + 7 * ldc + 8),
                            vcvt_f32_f16(vget_low_f16(v120_127))));
        vst1q_f32(c + 7 * ldc + 8 + 4,
                  vaddq_f32(vld1q_f32(c + 7 * ldc + 8 + 4),
                            vcvt_f32_f16(vget_high_f16(v120_127))));

        vst1q_f32(c + 7 * ldc + 16,
                  vaddq_f32(vld1q_f32(c + 7 * ldc + 16),
                            vcvt_f32_f16(vget_low_f16(v184_191))));
        vst1q_f32(c + 7 * ldc + 16 + 4,
                  vaddq_f32(vld1q_f32(c + 7 * ldc + 16 + 4),
                            vcvt_f32_f16(vget_high_f16(v184_191))));
      }
      c += 24;
      a -= 8 * K;
    }
    sc += ldc * 8;
    c = sc;
    a += 8 * K;
    b = sb;
  }
}
