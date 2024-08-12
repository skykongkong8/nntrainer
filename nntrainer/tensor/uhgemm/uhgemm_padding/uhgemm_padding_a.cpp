// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_padding_a.cpp
 * @date   05 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source file for padding function used in uhgemm
 *
 */

#include <arm_neon.h>
#include <uhgemm_padding_a.h>
#include <uhgemm_util.h>
#include <iostream>

void uhgemm_padding_A(const uint16_t *A, uint16_t *Ap, unsigned int M,
                     unsigned int K, unsigned int M8, unsigned int K8,
                     bool transA) {
  if (transA)
    uhgemm_padding_A_Trans(A, Ap, M, K, M8, K8);
  else
    uhgemm_padding_A_noTrans(A, Ap, M, K, M8, K8);
}

void uhgemm_padding_A_noTrans(const uint16_t *A, uint16_t *Ap, unsigned int M,
                             unsigned int K, unsigned int M8, unsigned int K8) {
  if (M != M8 && K != K8) {
    uhgemm_padding_A_noTrans_wrt_MK(A, Ap, M, K, M8, K8);
  } else if (M != M8) {
    uhgemm_padding_A_noTrans_wrt_M(A, Ap, M, K, M8, K8);
  } else if (K != K8) {
    uhgemm_padding_A_noTrans_wrt_K(A, Ap, M, K, M8, K8);
  } else {
    std::cerr << "Error : No room for matrix A padding!\n";
  }
}

void uhgemm_padding_A_Trans(const uint16_t *A, uint16_t *Ap, unsigned int M,
                           unsigned int K, unsigned int M8, unsigned int K8) {
  if (M != M8 && K != K8) {
    uhgemm_padding_A_Trans_wrt_MK(A, Ap, M, K, M8, K8);
  } else if (M != M8) {
    uhgemm_padding_A_Trans_wrt_M(A, Ap, M, K, M8, K8);
  } else if (K != K8) {
    uhgemm_padding_A_Trans_wrt_K(A, Ap, M, K, M8, K8);
  } else {
    std::cerr << "Error : No room for matrix A padding!\n";
  }
}

void uhgemm_padding_A_noTrans_wrt_M(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                   unsigned int K, unsigned int M8,
                                   unsigned int K8) {
  uint16x8_t ZEROS = vmovq_n_u16(0);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K; k += 8) {
      vst1q_u16(&Ap[m * K + k], vld1q_u16(&A[m * K + k]));
    }
  }
  for (unsigned int m = M; m < M8; ++m) {
    for (unsigned int k = 0; k < K; k += 8) {
      vst1q_u16(&Ap[m * K + k], ZEROS);
    }
  }
}

void uhgemm_padding_A_noTrans_wrt_K(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                   unsigned int K, unsigned int M8,
                                   unsigned int K8) {
  const unsigned int K8_low = (K >> 3) << 3;
  uint16x8_t ZEROS = vmovq_n_u16(0);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K8_low; k += 8) {
      vst1q_u16(&Ap[m * K8 + k], vld1q_u16(&A[m * K + k]));
    }
    for (unsigned int k = K8_low; k < K; ++k) {
      Ap[m * K8 + k] = A[m * K + k];
    }
    for (unsigned int k = K; k < K8; ++k) {
      Ap[m * K8 + k] = 0;
    }
  }
}

void uhgemm_padding_A_noTrans_wrt_MK(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                    unsigned int K, unsigned int M8,
                                    unsigned int K8) {
  const unsigned int K8_low = (K >> 3) << 3;
  uint16x8_t ZEROS = vmovq_n_u16(0);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K8_low; k += 8) {
      vst1q_u16(&Ap[m * K8 + k], vld1q_u16(&A[m * K + k]));
    }
    for (unsigned int k = K8_low; k < K; ++k) {
      Ap[m * K8 + k] = A[m * K + k];
    }
    for (unsigned int k = K; k < K8; ++k) {
      Ap[m * K8 + k] = 0;
    }
  }
  for (unsigned int m = M; m < M8; ++m) {
    for (unsigned int k = 0; k < K8; k += 8) {
      vst1q_u16(&Ap[m * K8 + k], ZEROS);
    }
  }
}

void uhgemm_padding_A_Trans_wrt_M(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                 unsigned int K, unsigned int M8,
                                 unsigned int K8) {
  const unsigned int M8_low = (M >> 3) << 3;
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int m = 0; m < M8_low; m += 8) {
      vst1q_u16(&Ap[k * M8 + m], vld1q_u16(&A[k * M + m]));
    }
    for (unsigned int m = M8_low; m < M; ++m) {
      Ap[k * M8 + m] = A[k * M + m];
    }
    for (unsigned int m = M; m < M8; ++m) {
      Ap[k * M8 + m] = 0;
    }
  }
}

void uhgemm_padding_A_Trans_wrt_K(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                 unsigned int K, unsigned int M8,
                                 unsigned int K8) {
  uint16x8_t ZEROS = vmovq_n_u16(0);
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int m = 0; m < M; m += 8) {
      vst1q_u16(&Ap[k * M8 + m], vld1q_u16(&A[k * M + m]));
    }
  }
  for (unsigned int k = K; k < K8; ++k) {
    for (unsigned int m = 0; m < M; m += 8) {
      vst1q_u16(&Ap[k * M8 + m], ZEROS);
    }
  }
}

void uhgemm_padding_A_Trans_wrt_MK(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                  unsigned int K, unsigned int M8,
                                  unsigned int K8) {
  uint16x8_t ZEROS = vmovq_n_u16(0);
  const unsigned int M8_low = (M >> 3) << 3;
  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int m = 0; m < M8_low; m += 8) {
      vst1q_u16(&Ap[k * M8 + m], vld1q_u16(&A[k * M + m]));
    }
    for (unsigned int m = M8_low; m < M; ++m) {
      Ap[k * M8 + m] = A[k * M + m];
    }
    for (unsigned int m = M; m < M8; ++m) {
      Ap[k * M8 + m] = 0;
    }
  }
  for (unsigned int k = K; k < K8; ++k) {
    for (unsigned int m = 0; m < M8; m += 8) {
      vst1q_u16(&Ap[k * M8 + m], ZEROS);
    }
  }
}
