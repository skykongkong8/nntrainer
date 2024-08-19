// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_util.cpp
 * @date   01 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is for util functions for half-precision GEMM
 */

#include <arm_neon.h>
#include <cmath>
#include <uhgemm_util.h>
#include <uhgemm_common.h>

/**
 * @brief aligned dynamic allocation function
 *
 * @param sz amount of data to allocate
 * @return uint16_t* addr of allocated memory
 */
uint16_t *alignedMalloc(unsigned int sz) {
  void *addr = 0;
  int iRet = posix_memalign(&addr, 64, sz * sizeof(uint16_t));
  assert(0 == iRet);
  return (uint16_t *)addr;
}

unsigned int get_next_mltpl_of_n(unsigned int x, unsigned int n) {
  assert(x > 0);
  return ((x - 1) / n + 1) * n;
}

unsigned int get_prev_mltpl_of_2p_n(unsigned int x, unsigned int n) {
  assert(x > 0);
  assert(n % 2 == 0);
  return (x >> n) << n;
}

void copy_C_to_C32(uint16_t *C, unsigned int *C32, unsigned int M, unsigned int N,
                   unsigned int beta) {
  uint32x4_t ZEROS = vmovq_n_u32(0);
  unsigned int size = M * N;
  unsigned int size4 = (size >> 2) << 2;
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);

  if (beta != 0) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N8_low; n += 8) {
        uint16x8_t c = vmulq_n_u16(vld1q_u16(&C[m * N + n]), beta);
        vst1q_u32(&C32[m * N + n], vmovl_u16(vget_low_u16(c)));
        vst1q_u32(&C32[m * N + n + 4], vmovl_u16(vget_high_u16(c)));
      }
      for (unsigned int n = N8_low; n < N; ++n) {
        C32[m * N + n] = beta * C[m * N + n];
      }
    }
  } else {
    for (unsigned int idx = 0; idx < size4; idx += 4) {
      vst1q_u32(&C32[idx], ZEROS);
    }
    for (unsigned int idx = size4; idx < size; idx++) {
      C32[idx] = 0;
    }
  }
}

void copy_C32_to_C(unsigned int *C32, uint16_t *C, unsigned int M, unsigned int N,
                   unsigned int beta) {
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      uint32x4_t x1 = vld1q_u32(&C32[m * N + n]);
      uint32x4_t x2 = vld1q_u32(&C32[m * N + n + 4]);
      vst1q_u16(&C[m * N + n],
                vcombine_u16(vmovn_u32(x1), vmovn_u32(x2)));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      C[m * N + n] = C32[m * N + n];
    }
  }
}
