/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include <arm_neon.h>

#include <cassert>
#include <cstdint>
#include <iostream>

#include "./mask_neon.h"

#define TRANSPOSE_FP32_4x4(row0, row1, row2, row3)                             \
  float32x4x2_t row01 = vtrnq_f32(row0, row1);                                 \
  float32x4x2_t row23 = vtrnq_f32(row2, row3);                                 \
  row0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])); \
  row1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])); \
  row2 =                                                                       \
    vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));    \
  row3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));

#define TRANSPOSE_FP16_4x4(row0, row1, row2, row3)                             \
  float16x4x2_t row01 = vtrn_f16(row0, row1);                                  \
  float16x4x2_t row23 = vtrn_f16(row2, row3);                                  \
  row0 = vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(row01.val[0])),   \
                                   vget_low_f32(vcvt_f32_f16(row23.val[0])))); \
  row1 = vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(row01.val[1])),   \
                                   vget_low_f32(vcvt_f32_f16(row23.val[1])))); \
  row2 =                                                                       \
    vcvt_f16_f32(vcombine_f32(vget_high_f32(vcvt_f32_f16(row01.val[0])),       \
                              vget_high_f32(vcvt_f32_f16(row23.val[0]))));     \
  row3 =                                                                       \
    vcvt_f16_f32(vcombine_f32(vget_high_f32(vcvt_f32_f16(row01.val[1])),       \
                              vget_high_f32(vcvt_f32_f16(row23.val[1]))));

// 4 * 4 = 16 instructions
static inline void transpose_kernel_4x4_neon(const float *src,
                                             unsigned int ld_src, float *dst,
                                             unsigned int ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3
  // b : b0 b1 b2 b3
  // c : c0 c1 c2 c3
  // d : d0 d1 d2 d3
  float32x4_t a = vld1q_f32(&src[0 * ld_src]);
  float32x4_t b = vld1q_f32(&src[1 * ld_src]);
  float32x4_t c = vld1q_f32(&src[2 * ld_src]);
  float32x4_t d = vld1q_f32(&src[3 * ld_src]);

  // transpose the 4x4 matrix formed by 32-bit elements: Macro from SSE
  // a : a0 b0 c0 d0
  // b : a1 b1 c1 d1
  // c : a2 b2 c2 d2
  // d : a3 b3 c3 d3
  TRANSPOSE_FP32_4x4(a, b, c, d);

  // store from registers to dst
  vst1q_f32(&dst[0 * ld_dst], a);
  vst1q_f32(&dst[1 * ld_dst], b);
  vst1q_f32(&dst[2 * ld_dst], c);
  vst1q_f32(&dst[3 * ld_dst], d);
}

// kernel for transpose mxn where m, n <= 4
// M + (M + 1) / 2 * 2 + 2 * N instructions
template <unsigned int M>
static void transpose_kernel_mxn_neon_128(unsigned int N, const float *src,
                                          unsigned int ld_src, float *dst,
                                          unsigned int ld_dst) {
  // clang-format off
  alignas(64) static const int masks[5][4] = {
    {  0,  0,  0,  0, },
    { -1,  0,  0,  0, },
    { -1, -1,  0,  0, },
    { -1, -1, -1,  0, },
    { -1, -1, -1, -1, },
  };
  // clang-format on

  // load from src to registers
  uint32x4_t mask_v = vld1q_u32(reinterpret_cast<const uint32_t *>(masks[N]));
  float32x4_t input[4];
  float32x4_t ZEROS = {0, 0, 0, 0};

  unsigned i;
  for (i = 0; i < M; ++i) {
    input[i] = vbslq_f32(mask_v, vld1q_f32(&src[i * ld_src]), ZEROS);
    // input[i] = _mm_maskload_ps(&src[i * ld_src], mask_v);
  }
  for (; i < 4; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = vmovq_n_f32(0.F);
  }

  float32x4_t temp[4];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = vzip1q_f32(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = vzip2q_f32(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 4; ++i) {
    temp[i] = vmovq_n_f32(0.F);
  }

  mask_v = vld1q_u32(reinterpret_cast<const uint32_t *>(masks[M]));
  for (i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      input[i] =
        vcombine_f32(vget_low_f32(temp[i / 2]), vget_low_f32(temp[2 + i / 2]));
    } else {
      input[i] = vcombine_f32(vget_high_f32(temp[2 + i / 2]),
                              vget_high_f32(temp[i / 2]));
    }
    vst1q_f32(&dst[i * ld_dst], vbslq_f32(mask_v, input[i], ZEROS));
  }
}

/* TOO MANY REGISTERS! 8x8 KERNEL : CANNOT BE DONE ON ARM ANYWAY ???
// 8 * 5 = 40 instructions
static inline void transpose_kernel_8x8_neon(const float *src,
                                             unsigned int ld_src, float *dst,
                                             unsigned int ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3 a4 a5 a6 a7
  // b : b0 b1 b2 b3 b4 b5 b6 b7
  // c : c0 c1 c2 c3 c4 c5 c6 c7
  // d : d0 d1 d2 d3 d4 d5 d6 d7
  // e : e0 e1 e2 e3 e4 e5 e6 e7
  // f : f0 f1 f2 f3 f4 f5 f6 f7
  // g : g0 g1 g2 g3 g4 g5 g6 g7
  // h : h0 h1 h2 h3 h4 h5 h6 h7

  // __m256 a = _mm256_loadu_ps(&src[0 * ld_src]);
  float32x4_t a0_3 = vld1q_f32(&src[0 * ld_src]);
  float32x4_t a4_7 = vld1q_f32(&src[0 * ld_src + 4]);
  // __m256 b = _mm256_loadu_ps(&src[1 * ld_src]);
  float32x4_t b0_3 = vld1q_f32(&src[1 * ld_src]);
  float32x4_t b4_7 = vld1q_f32(&src[1 * ld_src + 4]);
  // __m256 c = _mm256_loadu_ps(&src[2 * ld_src]);
  float32x4_t c0_3 = vld1q_f32(&src[2 * ld_src]);
  float32x4_t c4_7 = vld1q_f32(&src[2 * ld_src + 4]);
  // __m256 d = _mm256_loadu_ps(&src[3 * ld_src]);
  float32x4_t d0_3 = vld1q_f32(&src[3 * ld_src]);
  float32x4_t d4_7 = vld1q_f32(&src[3 * ld_src + 4]);
  // __m256 e = _mm256_loadu_ps(&src[4 * ld_src]);
  float32x4_t e0_3 = vld1q_f32(&src[4 * ld_src]);
  float32x4_t e4_7 = vld1q_f32(&src[4 * ld_src + 4]);
  // __m256 f = _mm256_loadu_ps(&src[5 * ld_src]);
  float32x4_t f0_3 = vld1q_f32(&src[5 * ld_src]);
  float32x4_t f4_7 = vld1q_f32(&src[5 * ld_src + 4]);
  // __m256 g = _mm256_loadu_ps(&src[6 * ld_src]);
  float32x4_t g0_3 = vld1q_f32(&src[6 * ld_src]);
  float32x4_t g4_7 = vld1q_f32(&src[6 * ld_src + 4]);
  // __m256 h = _mm256_loadu_ps(&src[7 * ld_src]);
  float32x4_t h0_3 = vld1q_f32(&src[7 * ld_src]);
  float32x4_t h4_7 = vld1q_f32(&src[7 * ld_src + 4]);

  __m256 ab0145, ab2367, cd0145, cd2367, ef0145, ef2367, gh0145, gh2367;
  __m256 ab0145, ab2367, cd0145, cd2367, ef0145, ef2367, gh0145, gh2367;
  __m256 abcd04, abcd15, efgh04, efgh15, abcd26, abcd37, efgh26, efgh37;
  __m256 abcd04, abcd15, efgh04, efgh15, abcd26, abcd37, efgh26, efgh37;
  // unpacking and interleaving 32-bit elements
  // ab0145 : a0 b0 a1 b1 a4 b4 a5 b5
  // ab2367 : a2 b2 a3 b3 a6 b6 a7 b7
  // cd0145 : c0 d0 c1 d1 c4 d4 c5 d5
  // cd2367 : c2 d2 c3 d3 c6 d6 c7 d7
  // ef0145 : e0 f0 e1 f1 e4 f4 e5 f5
  // ef2367 : e2 f2 e3 f3 e6 f6 e7 f7
  // gh0145 : g0 h0 g1 h1 g4 h4 g5 h5
  // gh2367 : g2 h2 g3 h3 g6 h6 g7 h7
  ab0145 = _mm256_unpacklo_ps(a, b);
  ab2367 = _mm256_unpackhi_ps(a, b);
  cd0145 = _mm256_unpacklo_ps(c, d);
  cd2367 = _mm256_unpackhi_ps(c, d);
  ef0145 = _mm256_unpacklo_ps(e, f);
  ef2367 = _mm256_unpackhi_ps(e, f);
  gh0145 = _mm256_unpacklo_ps(g, h);
  gh2367 = _mm256_unpackhi_ps(g, h);

  // shuffling the 32-bit elements
  // abcd04 : a0 b0 c0 d0 a4 b4 c4 d4
  // abcd15 : a1 b1 c1 d1 a5 b5 c5 d5
  // efgh04 : e0 f0 g0 h0 e4 f4 g4 h4
  // efgh15 : e1 f1 g1 h1 e5 b5 c5 d5
  // abcd26 : a2 b2 c2 d2 a6 b6 c6 d6
  // abcd37 : a3 b3 c3 d3 a7 b7 c7 d7
  // efgh26 : e2 f2 g2 h2 e6 f6 g6 h6
  // efgh37 : e3 f3 g3 h3 e7 f7 g7 h7
  abcd04 = _mm256_shuffle_ps(ab0145, cd0145, 0x44);
  abcd15 = _mm256_shuffle_ps(ab0145, cd0145, 0xee);
  efgh04 = _mm256_shuffle_ps(ef0145, gh0145, 0x44);
  efgh15 = _mm256_shuffle_ps(ef0145, gh0145, 0xee);
  abcd26 = _mm256_shuffle_ps(ab2367, cd2367, 0x44);
  abcd37 = _mm256_shuffle_ps(ab2367, cd2367, 0xee);
  efgh26 = _mm256_shuffle_ps(ef2367, gh2367, 0x44);
  efgh37 = _mm256_shuffle_ps(ef2367, gh2367, 0xee);

  // shuffling 128-bit elements
  // a : a0 b0 c0 d0 e0 f0 g0 h0
  // b : a1 b1 c1 d1 e1 f1 g1 h1
  // c : a2 b2 c2 d2 e2 f2 g2 h2
  // d : a3 b3 c3 d3 e3 f3 g3 h3
  // e : a4 b4 c4 d4 e4 f4 g4 h4
  // f : a5 b5 c5 d5 e5 f5 g5 h5
  // g : a6 b6 c6 d6 e6 f6 g6 h6
  // h : a7 b7 c7 d7 e7 f7 g7 h7
  a = _mm256_permute2f128_ps(efgh04, abcd04, 0x02);
  b = _mm256_permute2f128_ps(efgh15, abcd15, 0x02);
  c = _mm256_permute2f128_ps(efgh26, abcd26, 0x02);
  d = _mm256_permute2f128_ps(efgh37, abcd37, 0x02);
  e = _mm256_permute2f128_ps(efgh04, abcd04, 0x13);
  f = _mm256_permute2f128_ps(efgh15, abcd15, 0x13);
  g = _mm256_permute2f128_ps(efgh26, abcd26, 0x13);
  h = _mm256_permute2f128_ps(efgh37, abcd37, 0x13);

  // store from registers to dst
  _mm256_storeu_ps(&dst[0 * ld_dst], a);
  _mm256_storeu_ps(&dst[1 * ld_dst], b);
  _mm256_storeu_ps(&dst[2 * ld_dst], c);
  _mm256_storeu_ps(&dst[3 * ld_dst], d);
  _mm256_storeu_ps(&dst[4 * ld_dst], e);
  _mm256_storeu_ps(&dst[5 * ld_dst], f);
  _mm256_storeu_ps(&dst[6 * ld_dst], g);
  _mm256_storeu_ps(&dst[7 * ld_dst], h);
}

// kernel for transposing mxn where m, n <= 8
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + 2 * N instructions
template <unsigned int M>
static void transpose_kernel_mxn_neon_256(unsigned int N, const float *src,
                                          unsigned int ld_src, float *dst,
                                          unsigned int ld_dst) {
  // load from src to registers
  __m256i mask_v = _mm256_load_si256(
    reinterpret_cast<const __m256i *>(neon_ps_or_epi32_masks[N]));
  __m256 input[8];
  unsigned i;
  for (i = 0; i < M; ++i) {
    input[i] = _mm256_maskload_ps(&src[i * ld_src], mask_v);
  }
  for (; i < 8; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = _mm256_setzero_ps();
  }

  // unpacking and interleaving 32-bit elements
  __m256 temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm256_unpacklo_ps(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm256_unpackhi_ps(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = _mm256_setzero_ps();
  }

  // shuffling the 32-bit elements
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = _mm256_shuffle_ps(temp[4 * i], temp[4 * i + 2], 0x44);
    input[4 * i + 1] = _mm256_shuffle_ps(temp[4 * i], temp[4 * i + 2], 0xee);
    input[4 * i + 2] =
      _mm256_shuffle_ps(temp[4 * i + 1], temp[4 * i + 3], 0x44);
    input[4 * i + 3] =
      _mm256_shuffle_ps(temp[4 * i + 1], temp[4 * i + 3], 0xee);
  }

  // shuffling 128-bit elements
  // store from registers to dst
  mask_v = _mm256_load_si256(
    reinterpret_cast<const __m256i *>(neon_ps_or_epi32_masks[M]));
  for (i = 0; i < N; ++i) {
    if (i < 4) {
      temp[i] = _mm256_permute2f128_ps(input[4 + i], input[i], 0x02);
    } else {
      temp[i] = _mm256_permute2f128_ps(input[i], input[i - 4], 0x13);
    }
    _mm256_maskstore_ps(&dst[i * ld_dst], mask_v, temp[i]);
  }
}
*/

/* FP16 *////////////////////////////////////////////////////////////////////////////////

template <typename T>
static inline void print4(T v){
  for (int i = 0; i < 4; ++i){
    std::cout << v[i] << "\t";
  }
  std::cout << std::endl;
}

template <typename T>
static inline void print8(T v, int lim = 0){
  if (lim < 2){
  for (int i = 0; i < 8; ++i){
    std::cout << v[i] << "\t";
  }
  std::cout << std::endl;
  }
}

// 4 * 4 = 16 instructions
static inline void transpose_kernel_4x4_neon(const __fp16 *src,
                                             unsigned int ld_src, __fp16 *dst,
                                             unsigned int ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3
  // b : b0 b1 b2 b3
  // c : c0 c1 c2 c3
  // d : d0 d1 d2 d3
  float16x4_t a = vld1_f16(&src[0 * ld_src]);
  float16x4_t b = vld1_f16(&src[1 * ld_src]);
  float16x4_t c = vld1_f16(&src[2 * ld_src]);
  float16x4_t d = vld1_f16(&src[3 * ld_src]);

  // transpose the 4x4 matrix formed by 32-bit elements: Macro from SSE
  // a : a0 b0 c0 d0
  // b : a1 b1 c1 d1
  // c : a2 b2 c2 d2
  // d : a3 b3 c3 d3
  TRANSPOSE_FP16_4x4(a, b, c, d);

  // store from registers to dst
  vst1_f16(&dst[0 * ld_dst], a);
  vst1_f16(&dst[1 * ld_dst], b);
  vst1_f16(&dst[2 * ld_dst], c);
  vst1_f16(&dst[3 * ld_dst], d);
}

// kernel for transpose mxn where m, n <= 4
// M + (M + 1) / 2 * 2 + 2 * N instructions
template <unsigned int M>
static void transpose_kernel_mxn_neon_128(unsigned int N, const __fp16 *src,
                                          unsigned int ld_src, __fp16 *dst,
                                          unsigned int ld_dst) {
  // clang-format off
  alignas(64) static const int16_t masks[5][4] = {
    {  0,  0,  0,  0, },
    { -1,  0,  0,  0, },
    { -1, -1,  0,  0, },
    { -1, -1, -1,  0, },
    { -1, -1, -1, -1, },
  };
  // clang-format on

  // load from src to registers
  uint16x4_t mask_v = vld1_u16(reinterpret_cast<const uint16_t *>(masks[N]));
  float16x4_t input[4];
  float16x4_t ZEROS = vmov_n_f16(0.F);

  unsigned i;
  for (i = 0; i < M; ++i) {
    input[i] = vbsl_f16(mask_v, vld1_f16(&src[i * ld_src]), ZEROS);
  }
  for (; i < 4; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = vmov_n_f16(0.F);
  }

  float16x4_t temp[4];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = vzip1_f16(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = vzip2_f16(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 4; ++i) {
    temp[i] = vmov_n_f16(0.F);
  }

  mask_v = vld1_u16(reinterpret_cast<const uint16_t *>(masks[M]));
  for (i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      input[i] =
        vcvt_f16_f32(vcombine_f32(vget_low_f32(vcvt_f32_f16(temp[i / 2])),
                                  vget_low_f32(vcvt_f32_f16(temp[2 + i / 2]))));
    } else {
      input[i] =
        vcvt_f16_f32(vcombine_f32(vget_high_f32(vcvt_f32_f16(temp[i / 2])),
                                  vget_high_f32(vcvt_f32_f16(temp[2 + i / 2]))));
    }
    vst1_f16(&dst[i * ld_dst], vbsl_f16(mask_v, input[i], vld1_f16(&dst[i * ld_dst])));
    // vst1_f16(&dst[i * ld_dst], vbsl_f16(mask_v, input[i], ZEROS));
  }

}

// 8 * 5 = 40 instructions
static inline void transpose_kernel_8x8_neon(const __fp16 *src,
                                             unsigned int ld_src, __fp16 *dst,
                                             unsigned int ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3 a4 a5 a6 a7
  // b : b0 b1 b2 b3 b4 b5 b6 b7
  // c : c0 c1 c2 c3 c4 c5 c6 c7
  // d : d0 d1 d2 d3 d4 d5 d6 d7
  // e : e0 e1 e2 e3 e4 e5 e6 e7
  // f : f0 f1 f2 f3 f4 f5 f6 f7
  // g : g0 g1 g2 g3 g4 g5 g6 g7
  // h : h0 h1 h2 h3 h4 h5 h6 h7

  float16x8_t a = vld1q_f16(&src[0 * ld_src]);
  float16x8_t b = vld1q_f16(&src[1 * ld_src]);
  float16x8_t c = vld1q_f16(&src[2 * ld_src]);
  float16x8_t d = vld1q_f16(&src[3 * ld_src]);
  float16x8_t e = vld1q_f16(&src[4 * ld_src]);
  float16x8_t f = vld1q_f16(&src[5 * ld_src]);
  float16x8_t g = vld1q_f16(&src[6 * ld_src]);
  float16x8_t h = vld1q_f16(&src[7 * ld_src]);

  float16x8_t ab0145, ab2367, cd0145, cd2367, ef0145, ef2367, gh0145, gh2367;
  float16x8_t abcd04, abcd15, efgh04, efgh15, abcd26, abcd37, efgh26, efgh37;
  // unpacking and interleaving 32-bit elements
  // ab0145 : a0 b0 a1 b1 a4 b4 a5 b5
  // ab2367 : a2 b2 a3 b3 a6 b6 a7 b7
  // cd0145 : c0 d0 c1 d1 c4 d4 c5 d5
  // cd2367 : c2 d2 c3 d3 c6 d6 c7 d7
  // ef0145 : e0 f0 e1 f1 e4 f4 e5 f5
  // ef2367 : e2 f2 e3 f3 e6 f6 e7 f7
  // gh0145 : g0 h0 g1 h1 g4 h4 g5 h5
  // gh2367 : g2 h2 g3 h3 g6 h6 g7 h7
  ab0145 = vcombine_f16(vzip1_f16(vget_low_f16(a), vget_low_f16(b)),
                        vzip1_f16(vget_high_f16(a), vget_high_f16(b)));
  ab2367 = vcombine_f16(vzip2_f16(vget_low_f16(a), vget_low_f16(b)),
                        vzip2_f16(vget_high_f16(a), vget_high_f16(b)));
  cd0145 = vcombine_f16(vzip1_f16(vget_low_f16(c), vget_low_f16(d)),
                        vzip1_f16(vget_high_f16(c), vget_high_f16(d)));
  cd2367 = vcombine_f16(vzip2_f16(vget_low_f16(c), vget_low_f16(d)),
                        vzip2_f16(vget_high_f16(c), vget_high_f16(d)));
  ef0145 = vcombine_f16(vzip1_f16(vget_low_f16(e), vget_low_f16(f)),
                        vzip1_f16(vget_high_f16(e), vget_high_f16(f)));
  ef2367 = vcombine_f16(vzip2_f16(vget_low_f16(e), vget_low_f16(f)),
                        vzip2_f16(vget_high_f16(e), vget_high_f16(f)));
  gh0145 = vcombine_f16(vzip1_f16(vget_low_f16(g), vget_low_f16(h)),
                        vzip1_f16(vget_high_f16(g), vget_high_f16(h)));
  gh2367 = vcombine_f16(vzip2_f16(vget_low_f16(g), vget_low_f16(h)),
                        vzip2_f16(vget_high_f16(g), vget_high_f16(h)));

  // shuffling the 32-bit elements
  // abcd04 : a0 b0 c0 d0 a4 b4 c4 d4
  // abcd15 : a1 b1 c1 d1 a5 b5 c5 d5
  // efgh04 : e0 f0 g0 h0 e4 f4 g4 h4
  // efgh15 : e1 f1 g1 h1 e5 b5 c5 d5
  // abcd26 : a2 b2 c2 d2 a6 b6 c6 d6
  // abcd37 : a3 b3 c3 d3 a7 b7 c7 d7
  // efgh26 : e2 f2 g2 h2 e6 f6 g6 h6
  // efgh37 : e3 f3 g3 h3 e7 f7 g7 h7
  uint16x8_t shuffle_mask =
    vld1q_u16(reinterpret_cast<const uint16_t *>(shuffle_masks));
  abcd04 = vbslq_f16(shuffle_mask, ab0145, vextq_f16(cd0145, cd0145, 6));
  abcd15 = vbslq_f16(shuffle_mask, vextq_f16(ab0145, ab0145, 2), cd0145);
  
  efgh04 = vbslq_f16(shuffle_mask, ef0145, vextq_f16(gh0145, gh0145, 6));
  efgh15 = vbslq_f16(shuffle_mask, vextq_f16(ef0145, ef0145, 2), gh0145);

  abcd26 = vbslq_f16(shuffle_mask, ab2367, vextq_f16(cd2367, cd2367, 6));
  abcd37 = vbslq_f16(shuffle_mask, vextq_f16(ab2367, ab2367, 2), cd2367);

  efgh26 = vbslq_f16(shuffle_mask, ef2367, vextq_f16(gh2367, gh2367, 6));
  efgh37 = vbslq_f16(shuffle_mask, vextq_f16(ef2367, ef2367, 2), gh2367);

  // shuffling 128-bit elements
  // a : a0 b0 c0 d0 e0 f0 g0 h0
  // b : a1 b1 c1 d1 e1 f1 g1 h1
  // c : a2 b2 c2 d2 e2 f2 g2 h2
  // d : a3 b3 c3 d3 e3 f3 g3 h3
  // e : a4 b4 c4 d4 e4 f4 g4 h4
  // f : a5 b5 c5 d5 e5 f5 g5 h5
  // g : a6 b6 c6 d6 e6 f6 g6 h6
  // h : a7 b7 c7 d7 e7 f7 g7 h7
  a = vcombine_f16(vget_low_f16(abcd04), vget_low_f16(efgh04));
  b = vcombine_f16(vget_low_f16(abcd15), vget_low_f16(efgh15));
  c = vcombine_f16(vget_low_f16(abcd26), vget_low_f16(efgh26));
  d = vcombine_f16(vget_low_f16(abcd37), vget_low_f16(efgh37));
  e = vcombine_f16(vget_high_f16(abcd04), vget_high_f16(efgh04));
  f = vcombine_f16(vget_high_f16(abcd15), vget_high_f16(efgh15));
  g = vcombine_f16(vget_high_f16(abcd26), vget_high_f16(efgh26));
  h = vcombine_f16(vget_high_f16(abcd37), vget_high_f16(efgh37));

  // store from registers to dst
  vst1q_f16(&dst[0 * ld_dst], a);
  vst1q_f16(&dst[1 * ld_dst], b);
  vst1q_f16(&dst[2 * ld_dst], c);
  vst1q_f16(&dst[3 * ld_dst], d);
  vst1q_f16(&dst[4 * ld_dst], e);
  vst1q_f16(&dst[5 * ld_dst], f);
  vst1q_f16(&dst[6 * ld_dst], g);
  vst1q_f16(&dst[7 * ld_dst], h);
  // if (ld_src == 311 || ld_src == 821){
  //   print8(a);
  //   print8(b);
  //   print8(c);
  //   print8(d);
  //   print8(e);
  //   std::terminate();
  // }

}


// kernel for transposing mxn where m, n <= 8
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + 2 * N instructions
template <unsigned int M>
static void transpose_kernel_mxn_neon_256(unsigned int N, const __fp16 *src,
                                          unsigned int ld_src, __fp16 *dst,
                                          unsigned int ld_dst) {
  // load from src to registers
  float16x8_t ZEROS = vmovq_n_f16(0.F);
  uint16x8_t mask_v =
    vld1q_u16(reinterpret_cast<const uint16_t *>(neon_16bit_masks[N]));
  float16x8_t input[8];
  unsigned i;
// std::cout << "input\n";
  for (i = 0; i < M; ++i) {
// std::cout << src[i * ld_src] << "\t";
    input[i] = vbslq_f16(mask_v, vld1q_f16(&src[i * ld_src]), ZEROS);
    // print8(input[i], i);
  }
  // std::cout << "\nsrc end\n";
  for (; i < 8; ++i) {
    // Not really needed but to avoid uninitialized variable warning.
    // Shouldn't be much overhead because xor can be executed in parallel with
    // other instructions.
    input[i] = ZEROS;
  }
// std::cout << "temp\n";
  // unpacking and interleaving 32-bit elements
  float16x8_t temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = vcombine_f16(
      vzip1_f16(vget_low_f16(input[2 * i]), vget_low_f16(input[2 * i + 1])),
      vzip1_f16(vget_high_f16(input[2 * i]), vget_high_f16(input[2 * i + 1])));
    temp[2 * i + 1] = vcombine_f16(
      vzip2_f16(vget_low_f16(input[2 * i]), vget_low_f16(input[2 * i + 1])),
      vzip2_f16(vget_high_f16(input[2 * i]), vget_high_f16(input[2 * i + 1])));
    // print8(temp[2 * i], i);
    // print8(temp[2 * i + 1], i);
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = ZEROS;
  }

  // shuffling the 32-bit elements
  uint16x8_t shuffle_mask =
    vld1q_u16(reinterpret_cast<const uint16_t *>(shuffle_masks));
// std::cout << "input4\n";
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = vbslq_f16(shuffle_mask, temp[4 * i],
                             vextq_f16(temp[4 * i + 2], temp[4 * i + 2], 6));
    input[4 * i + 1] = vbslq_f16(
      shuffle_mask, vextq_f16(temp[4 * i], temp[4 * i], 2), temp[4 * i + 2]);
    input[4 * i + 2] =
      vbslq_f16(shuffle_mask, temp[4 * i + 1],
                vextq_f16(temp[4 * i + 3], temp[4 * i + 3], 6));
    input[4 * i + 3] =
      vbslq_f16(shuffle_mask, vextq_f16(temp[4 * i + 1], temp[4 * i + 1], 2),
                temp[4 * i + 3]);
    // print8(input[4 * i], i);
    // print8(input[4 * i + 1], i);
    // print8(input[4 * i + 2], i);
    // print8(input[4 * i + 3], i);
  }

  // shuffling 128-bit elements
  // store from registers to dst
// std::cout << "temp2\n";
  mask_v = vld1q_u16(
    reinterpret_cast<const uint16_t *>(neon_16bit_masks[M]));
  for (i = 0; i < N; ++i) {
    if (i < 4) {
      temp[i] = vcombine_f16(vget_low_f16(input[i]), vget_low_f16(input[4 + i]));
    // print8(temp[i], 0);
    } else {
      temp[i] = vcombine_f16(vget_high_f16(input[i - 4]), vget_high_f16(input[i])); 
    // print8(temp[i], 0);
    }
    vst1q_f16(&dst[i * ld_dst], vbslq_f16(mask_v, temp[i], vld1q_f16(&dst[i * ld_dst])));
    // vst1q_f16(&dst[i * ld_dst], vbslq_f16(mask_v, temp[i], ZEROS));

    // print8(vbslq_f16(mask_v, temp[i], ZEROS), i);
  }
}

#ifdef ENABLE_AVX

inline __m256i permute_row(__m256i row) {
  // clang-format off
  row = _mm256_shuffle_epi8(
      row,
      _mm256_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0,
                      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0));
  // clang-format on
  return row;
}

// template <>
inline static void transpose_kernel_8x32_neon(const uint8_t *src,
                                              unsigned int ld_src, uint8_t *dst,
                                              unsigned int ld_dst) {
  // load from src to registers
  // a : a0 a1 a2 a3 a4 a5 a6 a7 ... a31
  // b : b0 b1 b2 b3 b4 b5 b6 b7 ... b31
  // c : c0 c1 c2 c3 c4 c5 c6 c7 ... c31
  // d : d0 d1 d2 d3 d4 d5 d6 d7 ... d31
  // e : e0 e1 e2 e3 e4 e5 e6 e7 ... e31
  // f : f0 f1 f2 f3 f4 f5 f6 f7 ... f31
  // g : g0 g1 g2 g3 g4 g5 g6 g7 ... g31
  // h : h0 h1 h2 h3 h4 h5 h6 h7 ... h31

  // load from src
  __m256i a =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (0 * ld_src)));
  __m256i b =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (1 * ld_src)));
  __m256i c =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (2 * ld_src)));
  __m256i d =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (3 * ld_src)));
  __m256i e =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (4 * ld_src)));
  __m256i f =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (5 * ld_src)));
  __m256i g =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (6 * ld_src)));
  __m256i h =
    _mm256_loadu_si256(reinterpret_cast<const __m256i *>((src) + (7 * ld_src)));

  // shuffle in stride of one:
  // t0 : a0 -- a3,  b0 -- b3,  a4 -- a7, b4 -- b7,
  // a16 -- a19, b16 -- b19, a20 -- a23, b20 -- b23

  // t1 : a8 -- a11, b8 -- b11, a12 -- a15, b12 -- b15,
  // a24 -- a27, b24 -- b27, a28 -- a31, b28 -- b31

  // t2 : c0 -- c3,  d0 -- d3,  c4 -- c7, d4 -- d7,
  // c16 -- c19, d16 -- d19, c20 -- c23, d20 -- d23

  __m256i __t0 = _mm256_unpacklo_epi32(a, b);
  __m256i __t1 = _mm256_unpackhi_epi32(a, b);
  __m256i __t2 = _mm256_unpacklo_epi32(c, d);
  __m256i __t3 = _mm256_unpackhi_epi32(c, d);
  __m256i __t4 = _mm256_unpacklo_epi32(e, f);
  __m256i __t5 = _mm256_unpackhi_epi32(e, f);
  __m256i __t6 = _mm256_unpacklo_epi32(g, h);
  __m256i __t7 = _mm256_unpackhi_epi32(g, h);

  // shuffle in stride of two:
  // tt0: a0--a3, b0--b3, c0--c3, d0--d3,
  // a16--a19, b16 -- b19, c16 -- c19, d16--d19

  // tt1: a4 -- a7, b4 -- b7, c8--c11, d8--d11,
  // a20--a23, b20--b23, c20--c23, d20--d23

  // tt2: a8 -- a11, b8 -- b11, c8 -- c11, d8 -- d11,
  // a24 -- a27, b24 -- b27, c24 -- c27, d24 -- d27

  // tt3: a12 -- a15, b12 -- b15, c12--c15, d12--d15,
  // a28--a31, b28--b31, c28--c31, d28--d31

  // tt4:  e0--e3, f0--f3, g0--h3, g0--h3,
  // e16--e19, f16--f19, g16--h19, g16--h19
  __m256i __tt0 = _mm256_unpacklo_epi64(__t0, __t2);
  __m256i __tt1 = _mm256_unpackhi_epi64(__t0, __t2);
  __m256i __tt2 = _mm256_unpacklo_epi64(__t1, __t3);
  __m256i __tt3 = _mm256_unpackhi_epi64(__t1, __t3);
  __m256i __tt4 = _mm256_unpacklo_epi64(__t4, __t6);
  __m256i __tt5 = _mm256_unpackhi_epi64(__t4, __t6);
  __m256i __tt6 = _mm256_unpacklo_epi64(__t5, __t7);
  __m256i __tt7 = _mm256_unpackhi_epi64(__t5, __t7);

  // permute: pack consecutive elements(0-3) together
  // ttt0: a0--d0 a1--d1 a2--d2 a3--d3 a16-d16 a17-d17 a18-d18 a18-d19

  // ttt1: a4--d4 a5--d5 a6--d6 a7--d7 a20-d20 a21-d21 a22-d22 a23-d23

  // ttt2: a8--d8 a9--d9 a10--d10 a11--d11 a24-d24 a25-d25 a26-d26 a27-d27
  __m256i __ttt0 = permute_row(__tt0);
  __m256i __ttt1 = permute_row(__tt1);
  __m256i __ttt2 = permute_row(__tt2);
  __m256i __ttt3 = permute_row(__tt3);
  __m256i __ttt4 = permute_row(__tt4);
  __m256i __ttt5 = permute_row(__tt5);
  __m256i __ttt6 = permute_row(__tt6);
  __m256i __ttt7 = permute_row(__tt7);

  //
  // a: a0-h0 a1-h1 a16-h16 a17-h17
  // b: a2-h2 a3-h3 a18-h18 a19-h19

  // c: a4-h4 a6-h6 a20-h20 a22-h22 (a-h)x(4-7)
  // d: a5-h5 a7-h7 a21-h21 a23-h23 (a-h)x(20-23)

  // e: a8-h8 a9-h9 a24-h24 a25-h25 (a-h)x(8-11)
  // f: a10-h10 a11-h11 a26-h26 a27-h27 (a-h)x(24-27)

  // g: (a-h)x(12-15)
  // h: (a-h)x(28-31)
  a = _mm256_unpacklo_epi32(__ttt0, __ttt4);
  b = _mm256_unpackhi_epi32(__ttt0, __ttt4);
  c = _mm256_unpacklo_epi32(__ttt1, __ttt5);
  d = _mm256_unpackhi_epi32(__ttt1, __ttt5);
  e = _mm256_unpacklo_epi32(__ttt2, __ttt6);
  f = _mm256_unpackhi_epi32(__ttt2, __ttt6);
  g = _mm256_unpacklo_epi32(__ttt3, __ttt7);
  h = _mm256_unpackhi_epi32(__ttt3, __ttt7);

  // stores back 32 rows:

  reinterpret_cast<uint64_t *>(dst)[0] = _mm256_extract_epi64(a, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst)[0] = _mm256_extract_epi64(a, 1);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 2)[0] =
    _mm256_extract_epi64(b, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 3)[0] =
    _mm256_extract_epi64(b, 1);

  reinterpret_cast<uint64_t *>((dst) + ld_dst * 4)[0] =
    _mm256_extract_epi64(c, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 5)[0] =
    _mm256_extract_epi64(c, 1);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 6)[0] =
    _mm256_extract_epi64(d, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 7)[0] =
    _mm256_extract_epi64(d, 1);

  reinterpret_cast<uint64_t *>((dst) + ld_dst * 8)[0] =
    _mm256_extract_epi64(e, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 9)[0] =
    _mm256_extract_epi64(e, 1);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 10)[0] =
    _mm256_extract_epi64(f, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 11)[0] =
    _mm256_extract_epi64(f, 1);

  reinterpret_cast<uint64_t *>((dst) + ld_dst * 12)[0] =
    _mm256_extract_epi64(g, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 13)[0] =
    _mm256_extract_epi64(g, 1);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 14)[0] =
    _mm256_extract_epi64(h, 0);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 15)[0] =
    _mm256_extract_epi64(h, 1);

  reinterpret_cast<uint64_t *>((dst) + ld_dst * 16)[0] =
    _mm256_extract_epi64(a, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 17)[0] =
    _mm256_extract_epi64(a, 3);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 18)[0] =
    _mm256_extract_epi64(b, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 19)[0] =
    _mm256_extract_epi64(b, 3);

  reinterpret_cast<uint64_t *>((dst) + ld_dst * 20)[0] =
    _mm256_extract_epi64(c, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 21)[0] =
    _mm256_extract_epi64(c, 3);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 22)[0] =
    _mm256_extract_epi64(d, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 23)[0] =
    _mm256_extract_epi64(d, 3);

  reinterpret_cast<uint64_t *>((dst) + ld_dst * 24)[0] =
    _mm256_extract_epi64(e, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 25)[0] =
    _mm256_extract_epi64(e, 3);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 26)[0] =
    _mm256_extract_epi64(f, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 27)[0] =
    _mm256_extract_epi64(f, 3);

  reinterpret_cast<uint64_t *>((dst) + ld_dst * 28)[0] =
    _mm256_extract_epi64(g, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 29)[0] =
    _mm256_extract_epi64(g, 3);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 30)[0] =
    _mm256_extract_epi64(h, 2);
  reinterpret_cast<uint64_t *>((dst) + ld_dst * 31)[0] =
    _mm256_extract_epi64(h, 3);
}

static inline void load_with_remainders_i16(const uint16_t *src,
                                            unsigned int ld_src, __m256i r[],
                                            unsigned mrem, unsigned nrem) {
  if (nrem < 16) {
    uint16_t local_buffer[16] = {0};
    __m256i mask_nrem_v = _mm256_load_si256(
      reinterpret_cast<const __m256i *>(neon_ps_or_epi32_masks[nrem / 2]));
    unsigned half = nrem % 2;
    for (unsigned i = 0; i < mrem; ++i) {
      // mask load
      r[i] = _mm256_maskload_epi32(
        reinterpret_cast<const int *>(&src[i * ld_src]), mask_nrem_v);
      if (half == 1) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(&local_buffer[0]),
                            r[i]);
        local_buffer[nrem - 1] = src[i * ld_src + nrem - 1];
        r[i] = _mm256_loadu_si256(
          reinterpret_cast<const __m256i *>(&local_buffer[0]));
      }
    }
  } else {
    for (unsigned i = 0; i < mrem; ++i) {
      // normal load
      r[i] =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i * ld_src));
    }
  }
}

static inline void store_with_remainders_i16(uint16_t *dst, unsigned int ld_dst,
                                             __m256i u[], unsigned mrem,
                                             unsigned nrem) {
  if (mrem < 8) {
    uint16_t local_buffer[8] = {0};
    __m256i mask_mrem_v = _mm256_load_si256(
      reinterpret_cast<const __m256i *>(neon_ps_or_epi32_masks[mrem / 2]));
    unsigned half = mrem % 2;
    unsigned i = 0;
    for (; i < nrem; i += 1) {
      // mask store
      int reg_idx = i % 8;
      __m128i d;
      if (i >= 8) {
        d = _mm256_extractf128_si256(u[reg_idx], 1);
      } else {
        d = _mm256_extractf128_si256(u[reg_idx], 0);
      }
      _mm256_maskstore_epi32(reinterpret_cast<int *>(dst + i * ld_dst),
                             mask_mrem_v, _mm256_castsi128_si256(d));
      if (half == 1) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(local_buffer), d);
        (dst + i * ld_dst)[mrem - 1] = local_buffer[mrem - 1];
      }
    }

  } else {
    unsigned i = 0;
    for (; i < nrem; i += 1) {
      // normal store
      unsigned reg_idx = i % 8;
      if (i >= 8) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i * ld_dst),
                         _mm256_extractf128_si256(u[reg_idx], 1));
      } else {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + i * ld_dst),
                         _mm256_extractf128_si256(u[reg_idx], 0));
      }
    }
  }
}

template <bool MREM = false, bool NREM = false>
inline static void
transpose_kernel_8x16_neon(const uint16_t *src, unsigned int ld_src,
                           uint16_t *dst, unsigned int ld_dst,
                           unsigned mrem = 8, unsigned nrem = 16) {
  __m256i r[8];
  // load from src to registers
  // a : a0 a1 a2 a3 a4 a5 a6 a7 ... a15
  // b : b0 b1 b2 b3 b4 b5 b6 b7 ... b15
  // c : c0 c1 c2 c3 c4 c5 c6 c7 ... c15
  // d : d0 d1 d2 d3 d4 d5 d6 d7 ... d15
  // e : e0 e1 e2 e3 e4 e5 e6 e7 ... e15
  // f : f0 f1 f2 f3 f4 f5 f6 f7 ... f15
  // g : g0 g1 g2 g3 g4 g5 g6 g7 ... g15
  // h : h0 h1 h2 h3 h4 h5 h6 h7 ... h15
  if (MREM || NREM) {
    load_with_remainders_i16(src, ld_src, r, mrem, nrem);
  } else {
    r[0] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (0 * ld_src)));
    r[1] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (1 * ld_src)));
    r[2] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (2 * ld_src)));
    r[3] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (3 * ld_src)));
    r[4] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (4 * ld_src)));
    r[5] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (5 * ld_src)));
    r[6] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (6 * ld_src)));
    r[7] = _mm256_loadu_si256(
      reinterpret_cast<const __m256i *>((src) + (7 * ld_src)));
  }
  // t0 : a0a1, b0b1, a2a3, b2b3,
  // a8a9, b8b9, a10a11, b10b11

  // t1 : a4a5, b4b5, a6a7, b6b7,
  // a12a13, b12b13, a14a15, b14b15

  // t2 : c0c1, d0d1, c2c3, d2d3,
  // c8c9, d8d9, c10c11, d10d11

  __m256i __t0 = _mm256_unpacklo_epi32(r[0], r[1]);
  __m256i __t1 = _mm256_unpackhi_epi32(r[0], r[1]);
  __m256i __t2 = _mm256_unpacklo_epi32(r[2], r[3]);
  __m256i __t3 = _mm256_unpackhi_epi32(r[2], r[3]);
  __m256i __t4 = _mm256_unpacklo_epi32(r[4], r[5]);
  __m256i __t5 = _mm256_unpackhi_epi32(r[4], r[5]);
  __m256i __t6 = _mm256_unpacklo_epi32(r[6], r[7]);
  __m256i __t7 = _mm256_unpackhi_epi32(r[6], r[7]);

  // tt0: a0a1, b0b1, c0c1, d0d1,
  // a9a9, b8b9, c8c9, d8d9

  // tt1: a2a3, b2b3, c2c3, d2d3,
  // a10a11, b10b11, c10c11, d10d11

  // tt2: a4a5, b4b5, c4c5, d4d5,
  // a12a13, b12b13, c12c13, d12d13

  // tt3: a6a7, b6b7, c6c7, d6d7,
  // a14a15, b14b15, c14c15, d14d15

  // tt4: e0e1, f0f1, g0g1, h0h1,
  // e9e9, f8f9, g8g9, h8h9
  __m256i __tt0 = _mm256_unpacklo_epi64(__t0, __t2);
  __m256i __tt1 = _mm256_unpackhi_epi64(__t0, __t2);
  __m256i __tt2 = _mm256_unpacklo_epi64(__t1, __t3);
  __m256i __tt3 = _mm256_unpackhi_epi64(__t1, __t3);
  __m256i __tt4 = _mm256_unpacklo_epi64(__t4, __t6);
  __m256i __tt5 = _mm256_unpackhi_epi64(__t4, __t6);
  __m256i __tt6 = _mm256_unpacklo_epi64(__t5, __t7);
  __m256i __tt7 = _mm256_unpackhi_epi64(__t5, __t7);

  // t0: a0b0, a1b1, c0c1, d0d1,
  // a8b8, a9b9, c8c9, d8d9
  __t0 = _mm256_shufflelo_epi16(__tt0, 0xD8);
  __t1 = _mm256_shufflelo_epi16(__tt1, 0xD8);
  __t2 = _mm256_shufflelo_epi16(__tt2, 0xD8);
  __t3 = _mm256_shufflelo_epi16(__tt3, 0xD8);
  __t4 = _mm256_shufflelo_epi16(__tt4, 0xD8);
  __t5 = _mm256_shufflelo_epi16(__tt5, 0xD8);
  __t6 = _mm256_shufflelo_epi16(__tt6, 0xD8);
  __t7 = _mm256_shufflelo_epi16(__tt7, 0xD8);

  // tt0: a0b0, a1b1, c0d0, c1d1,
  // a8b8, a9b9, c8d8, c9d9
  __tt0 = _mm256_shufflehi_epi16(__t0, 0xD8);
  __tt1 = _mm256_shufflehi_epi16(__t1, 0xD8);
  __tt2 = _mm256_shufflehi_epi16(__t2, 0xD8);
  __tt3 = _mm256_shufflehi_epi16(__t3, 0xD8);
  __tt4 = _mm256_shufflehi_epi16(__t4, 0xD8);
  __tt5 = _mm256_shufflehi_epi16(__t5, 0xD8);
  __tt6 = _mm256_shufflehi_epi16(__t6, 0xD8);
  __tt7 = _mm256_shufflehi_epi16(__t7, 0xD8);

  // t0: a0b0, c0d0, a1b1, c1d1,
  // a8b8, c8d8, a9b9, c9d9
  __t0 = _mm256_shuffle_epi32(__tt0, 0xD8);
  __t1 = _mm256_shuffle_epi32(__tt1, 0xD8);
  __t2 = _mm256_shuffle_epi32(__tt2, 0xD8);
  __t3 = _mm256_shuffle_epi32(__tt3, 0xD8);
  // t4: e0f0, g0h0, e1f1, g1h1,
  // e8f8, g8h8, e9f9, g9h9
  __t4 = _mm256_shuffle_epi32(__tt4, 0xD8);
  __t5 = _mm256_shuffle_epi32(__tt5, 0xD8);
  __t6 = _mm256_shuffle_epi32(__tt6, 0xD8);
  __t7 = _mm256_shuffle_epi32(__tt7, 0xD8);

  // r0: a0b0, c0d0, e0f0, g0h0,
  // a8b8, c8d8, e8f8, g8h8
  r[0] = _mm256_unpacklo_epi64(__t0, __t4); // 0, 8
  // r1: a1b1, c1d1, e1f1, g1h1,
  // a9b9, c9d9, e9f9, g9h9
  r[1] = _mm256_unpackhi_epi64(__t0, __t4); // 1, 9
  r[2] = _mm256_unpacklo_epi64(__t1, __t5); // 2, 10
  r[3] = _mm256_unpackhi_epi64(__t1, __t5); // 3, 11
  r[4] = _mm256_unpacklo_epi64(__t2, __t6); // 4, 12
  r[5] = _mm256_unpackhi_epi64(__t2, __t6); // 5, 13
  r[6] = _mm256_unpacklo_epi64(__t3, __t7); // 6, 14
  r[7] = _mm256_unpackhi_epi64(__t3, __t7); // 7, 15

  // stores back 16 rows:
  if (MREM || NREM) {
    store_with_remainders_i16(dst, ld_dst, r, mrem, nrem);
  } else {
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst),
                     _mm256_extractf128_si256(r[0], 0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst),
                     _mm256_extractf128_si256(r[1], 0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 2),
                     _mm256_extractf128_si256(r[2], 0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 3),
                     _mm256_extractf128_si256(r[3], 0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 4),
                     _mm256_extractf128_si256(r[4], 0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 5),
                     _mm256_extractf128_si256(r[5], 0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 6),
                     _mm256_extractf128_si256(r[6], 0));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 7),
                     _mm256_extractf128_si256(r[7], 0));

    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 8),
                     _mm256_extractf128_si256(r[0], 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 9),
                     _mm256_extractf128_si256(r[1], 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 10),
                     _mm256_extractf128_si256(r[2], 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 11),
                     _mm256_extractf128_si256(r[3], 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 12),
                     _mm256_extractf128_si256(r[4], 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 13),
                     _mm256_extractf128_si256(r[5], 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 14),
                     _mm256_extractf128_si256(r[6], 1));
    _mm_storeu_si128(reinterpret_cast<__m128i *>((dst) + ld_dst * 15),
                     _mm256_extractf128_si256(r[7], 1));
  }
}

// kernel for transposing mxn where m, n <= 8
// M + (M + 1) / 2 * 2 + (M + 3) / 4 * 4 + 2 * N instructions
template <unsigned int M>
static void transpose_kernel_mxn_neon_uint8(unsigned int N, const uint8_t *src,
                                            unsigned int ld_src, uint8_t *dst,
                                            unsigned int ld_dst) {
  // load from src to registers
  // first load masks
  __m256i mask_v = _mm256_load_si256(
    reinterpret_cast<const __m256i *>(neon_ps_or_epi32_masks[N / 4]));

  __m256i input[8] = {0};
  //   __m256i input[8];
  unsigned i, j;
  for (i = 0; i < M; ++i) {
    uint8_t local_buffer[32] = {0};

    // first load into local buffer with mask
    input[i] = _mm256_maskload_epi32(
      reinterpret_cast<const int *>(src + i * ld_src), mask_v);

    _mm256_storeu_si256(reinterpret_cast<__m256i *>(&local_buffer[0]),
                        input[i]);

    // fill in the local buffer with the remainder elements
    for (j = N / 4 * 4; j < N; j++)
      local_buffer[j] = src[i * ld_src + j];

    // from local buffer to input registers
    input[i] =
      _mm256_loadu_si256(reinterpret_cast<__m256i *>(&local_buffer[0]));
  }

  // for (; i < 8; ++i) {
  // input[i] = _mm256_setzero_si256();
  //}

  // interleaving 8-bit elements
  // e.g., temp[0] now becomes: a0 b0 a1 b1 a2 b2 ...
  __m256i temp[8];
  for (i = 0; i < (M + 1) / 2; ++i) {
    temp[2 * i] = _mm256_unpacklo_epi8(input[2 * i], input[2 * i + 1]);
    temp[2 * i + 1] = _mm256_unpackhi_epi8(input[2 * i], input[2 * i + 1]);
  }
  for (i = i * 2; i < 8; ++i) {
    temp[i] = _mm256_setzero_si256();
  }

  // interleaving 16-bit elements
  // e.g., temp[0] now becomes: a0 b0 c0 d0 a1 b1 c1 d1 ...
  for (i = 0; i < (M + 3) / 4; ++i) {
    input[4 * i] = _mm256_unpacklo_epi16(temp[i * 4], temp[i * 4 + 2]);
    input[4 * i + 1] = _mm256_unpackhi_epi16(temp[i * 4], temp[i * 4 + 2]);
    input[4 * i + 2] = _mm256_unpacklo_epi16(temp[i * 4 + 1], temp[i * 4 + 3]);
    input[4 * i + 3] = _mm256_unpackhi_epi16(temp[i * 4 + 1], temp[i * 4 + 3]);
  }

  // interleaving 32-bit elements
  // e.g., temp[0] now becomes a0 b0 c0 d0 e0 f0 g0 h0 ...
  for (i = 0; i < 4 /*(M + 1) / 2*/; ++i) {
    temp[2 * i] = _mm256_unpacklo_epi32(input[i], input[(i + 4)]);
    temp[2 * i + 1] = _mm256_unpackhi_epi32(input[i], input[(i + 4)]);
  }

  // retrieve the final result, extract every 64-bit
  // i.e., take a 256-bit temp[0] for example, that will
  // 0-63 bit: a0 -- h0,
  // 64-127 bit: a1 -- h1,
  // 128-191 bit:  a16 -- h16,
  // 192-255 bit:   a17 -- h17
  uint64_t t;
  mask_v = _mm256_load_si256(
    reinterpret_cast<const __m256i *>(neon_ps_or_epi32_masks[M / 4]));
  for (i = 0; i < N; ++i) {
    if (i < 16) {
      if (i % 2 == 0)
        t = _mm256_extract_epi64(temp[i / 2], 0);
      else
        t = _mm256_extract_epi64(temp[i / 2], 1);

    } else {
      if (i % 2 == 0)
        t = _mm256_extract_epi64(temp[(i - 16) / 2], 2);
      else
        t = _mm256_extract_epi64(temp[(i - 16) / 2], 3);
    }
    __m256i t_vec = _mm256_set_epi64x(0, 0, 0, t);
    _mm256_maskstore_epi32(reinterpret_cast<int *>(dst + i * ld_dst), mask_v,
                           t_vec);
    for (j = M / 4 * 4; j < M; j++) {
      dst[ld_dst * i + j] = ((t >> (8 * j)) & 255);
    }
  }
}

// #endif // __AVX2__
#endif
