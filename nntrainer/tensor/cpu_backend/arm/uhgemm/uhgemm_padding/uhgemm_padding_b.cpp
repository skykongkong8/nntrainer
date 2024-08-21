// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_padding_b.cpp
 * @date   05 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source file for padding function used in uhgemm
 *
 */

#include <arm_neon.h>
#include <uhgemm_padding_b.h>
#include <uhgemm_util.h>
#include <iostream>

void uhgemm_padding_B(const uint16_t *B, uint16_t *Bp, unsigned int K,
                     unsigned int N, unsigned int K8, unsigned int N16,
                     bool transB) {
  if (transB) {
    return uhgemm_padding_B_Trans(B, Bp, K, N, K8, N16);
  } else {
    return uhgemm_padding_B_noTrans(B, Bp, K, N, K8, N16);
  }
}

void uhgemm_padding_B_noTrans(const uint16_t *B, uint16_t *Bp, unsigned int K,
                             unsigned int N, unsigned int K8,
                             unsigned int N16) {
  if (K != K8 && N != N16) {
    return uhgemm_padding_B_noTrans_wrt_KN(B, Bp, K, N, K8, N16);
  } else if (K != K8) {
    return uhgemm_padding_B_noTrans_wrt_K(B, Bp, K, N, K8, N16);
  } else if (N != N16) {
    return uhgemm_padding_B_noTrans_wrt_N(B, Bp, K, N, K8, N16);
  } else {
    std::cerr << "Error : No room for matrix B padding\n";
  }
}

void uhgemm_padding_B_Trans(const uint16_t *B, uint16_t *Bp, unsigned int K,
                           unsigned int N, unsigned int K8, unsigned int N16) {
  if (K != K8 && N != N16) {
    return uhgemm_padding_B_Trans_wrt_KN(B, Bp, K, N, K8, N16);
  } else if (K != K8) {
    return uhgemm_padding_B_Trans_wrt_K(B, Bp, K, N, K8, N16);
  } else if (N != N16) {
    return uhgemm_padding_B_Trans_wrt_N(B, Bp, K, N, K8, N16);
  } else {
    std::cerr << "Error : No room for matrix B padding\n";
  }
}

void uhgemm_padding_B_noTrans_wrt_N(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                   unsigned int N, unsigned int K8,
                                   unsigned int N16) {
  const unsigned int N8_low = (N >> 3) << 3;
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      vst1q_u16(&Bp[k * N16 + n], vld1q_u16(&B[k * N + n]));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      Bp[k * N16 + n] = B[k * N + n];
    }
    for (unsigned int n = N; n < N16; ++n) {
      Bp[k * N16 + n] = 0;
    }
  }
}

void uhgemm_padding_B_noTrans_wrt_K(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                   unsigned int N, unsigned int K8,
                                   unsigned int N16) {
  uint16x8_t ZEROS = vmovq_n_u16(0);

  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; n += 8) {
      vst1q_u16(&Bp[k * N + n], vld1q_u16(&B[k * N + n]));
    }
  }
  for (unsigned int k = K; k < K8; ++k) {
    for (unsigned int n = 0; n < N; n += 8) {
      vst1q_u16(&Bp[k * N + n], ZEROS);
    }
  }
}

void uhgemm_padding_B_noTrans_wrt_KN(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                    unsigned int N, unsigned int K8,
                                    unsigned int N16) {
  unsigned int N8_low = (N >> 3) << 3;
  uint16x8_t ZEROS = vmovq_n_u16(0);
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      vst1q_u16(&Bp[k * N16 + n], vld1q_u16(&B[k * N + n]));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      Bp[k * N16 + n] = B[k * N + n];
    }
    for (unsigned int n = N; n < N16; ++n) {
      Bp[k * N16 + n] = 0;
    }
  }
  for (unsigned int k = K; k < K8; ++k) {
    for (unsigned int n = 0; n < N16; n += 8) {
      vst1q_u16(&Bp[k * N16 + n], ZEROS);
    }
  }
}

void uhgemm_padding_B_Trans_wrt_N(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                 unsigned int N, unsigned int K8,
                                 unsigned int N16) {
  uint16x8_t ZEROS = vmovq_n_u16(0);

  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K; k += 8) {
      vst1q_u16(&Bp[n * K8 + k], vld1q_u16(&B[n * K + k]));
    }
  }
  for (unsigned int n = N; n < N16; ++n) {
    for (unsigned int k = 0; k < K; k += 8) {
      vst1q_u16(&Bp[n * K8 + k], ZEROS);
    }
  }
}

void uhgemm_padding_B_Trans_wrt_K(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                 unsigned int N, unsigned int K8,
                                 unsigned int N16) {
  const unsigned int K8_low = (K >> 3) << 3;
  uint16x8_t ZEROS = vmovq_n_u16(0);

  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K8_low; k += 8) {
      vst1q_u16(&Bp[n * K8 + k], vld1q_u16(&B[n * K + k]));
    }
    for (unsigned int k = K8_low; k < K; ++k) {
      Bp[n * K8 + k] = B[n * K + k];
    }
    for (unsigned int k = K; k < K8; ++k) {
      Bp[n * K8 + k] = 0;
    }
  }
}

void uhgemm_padding_B_Trans_wrt_KN(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                  unsigned int N, unsigned int K8,
                                  unsigned int N16) {
  unsigned int K8_low = (K >> 3) << 3;
  uint16x8_t ZEROS = vmovq_n_u16(0);

  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K8_low; k += 8) {
      vst1q_u16(&Bp[n * K8 + k], vld1q_u16(&B[n * K + k]));
    }
    for (unsigned int k = K8_low; k < K; ++k) {
      Bp[n * K8 + k] = B[n * K + k];
    }
    for (unsigned int k = K; k < K8; ++k) {
      Bp[n * K8 + k] = 0;
    }
  }
  for (unsigned int n = N; n < N16; ++n) {
    for (unsigned int k = 0; k < K8; k += 8) {
      vst1q_u16(&Bp[n * K8 + k], ZEROS);
    }
  }
}
