// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_padding_a.cpp
 * @date   05 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source file for padding function used in hgemm
 *
 */

#include <arm_neon.h>
#include <hgemm_padding_a.h>
#include <hgemm_util.h>
#include <iostream>
#include <stdexcept>

void hgemm_padding_A(const __fp16 *A, const __fp16 *Ap, unsigned int M,
                     unsigned int K, unsigned int M8, unsigned int K8,
                     bool transA) {
  if (transA)
    hgemm_padding_A_Trans(A, Ap, M, K, M8, K8);
  else
    hgemm_padding_A_noTrans(A, Ap, M, K, M8, K8);
}

void hgemm_padding_A_noTrans(const __fp16 *A, const __fp16 *Ap, unsigned int M,
                             unsigned int K, unsigned int M8, unsigned int K8) {
  if (M != M8 && K != K8) {
    hgemm_padding_A_noTrans_wrt_MK(A, Ap, M, K, M8, K8);
  } else if (M != M8) {
    hgemm_padding_A_noTrans_wrt_M(A, Ap, M, K, M8, K8);
  } else if (K != K8) {
    hgemm_padding_A_noTrans_wrt_K(A, Ap, M, K, M8, K8);
  } else {
    std::invalid_argument("Error : No room for matrix A padding");
  }
}

void hgemm_padding_A_noTrans_wrt_M(const __fp16 *A, const __fp16 *Ap,
                                   unsigned int M, unsigned int K,
                                   unsigned int M8, unsigned int K8) {
  float16x8_t ZEROS = vmovq_n_f16(0.F);

  __fp16 *A8 = (__fp16 *)Ap;

  // padding for M
  A8 = alignedMalloc(M8 * K);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K; k += 8) {
      vst1q_f16(&A8[m * K + k], vld1q_f16(&A8[m * K + k]));
    }
  }
  for (unsigned int m = 0; m < M8; ++m) {
    for (unsigned int k = 0; k < K; k += 8) {
      vst1q_f16(&A8[m * K + k], ZEROS);
    }
  }
}

void hgemm_padding_A_noTrans_wrt_K(const __fp16 *A, const __fp16 *Ap,
                                   unsigned int M, unsigned int K,
                                   unsigned int M8, unsigned int K8) {
  const unsigned int K8_low = (K >> 3) << 3;
  float16x8_t ZEROS = vmovq_n_f16(0.F);

  __fp16 *A8 = (__fp16 *)Ap;
  A8 = alignedMalloc(M * K8);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K8_low; k += 8) {
      vst1q_f16(&A8[m * K8 + k], vld1q_f16(&A[m * K + k]));
    }
    for (unsigned int k = K8_low; k < K; ++k) {
      A8[m * K8 + k] = A[m * K + k];
    }
    for (unsigned int k = K; k < K8; ++k) {
      A8[m * K8 + k] = 0.F;
    }
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int k = 0; k < K; ++k) {
      if (A8[m * K8 + k] != A[m * K + k])
        std::cout << "FATAL A ERROR!\n";
    }
    for (unsigned int k = K; k < K8; ++k) {
      if (A8[m * K8 + k] != 0)
        std::cout << "FATAL non-zero A ERROR!\n";
    }
  }
/*MATRIX PRINT DEBUG*/
//   std::cout << "A Matrix\n";
//   for (unsigned int m = 0; m < M; ++m) {
//     for (unsigned int k = 0; k < K; ++k) {
//       std::cout << A[m * K + k] << "\t";
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;

//   std::cout << "padded A Matrix\n";
//   for (unsigned int m = 0; m < M; ++m) {
//     for (unsigned int k = 0; k < K8; ++k) {
//       std::cout << A8[m * K8 + k] << "\t";
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;
}

void hgemm_padding_A_noTrans_wrt_MK(const __fp16 *A, const __fp16 *Ap,
                                    unsigned int M, unsigned int K,
                                    unsigned int M8, unsigned int K8) {
  std::invalid_argument("NYI : hgemm_padding_A_noTrans_wrt_MK");
}

void hgemm_padding_A_Trans(const __fp16 *A, const __fp16 *Ap, unsigned int M,
                           unsigned int K, unsigned int M8, unsigned int K8) {
  std::invalid_argument("NYI : hgemm_padding_A_Trans");
}

void hgemm_padding_A_Trans_wrt_M(const __fp16 *A, const __fp16 *Ap,
                                 unsigned int M, unsigned int K,
                                 unsigned int M8, unsigned int K8) {
  const unsigned int M8_high = ((M - 1) / 8 + 1) * 8;
  const unsigned int M8_low = (M >> 3) << 3;
  __fp16 *A8 = (__fp16 *)Ap;

  A8 = alignedMalloc(M8_high * K);

  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int m = 0; m < M8_low; m += 8) {
      vst1q_f16(&A8[k * M + m], vld1q_f16(&A[k * M + m]));
    }
    for (unsigned int m = M8_low; m < M; ++m) {
      A8[k * M + m] = A[k * M + m];
    }
    for (unsigned int m = M; m < M8_high; ++m) {
      A8[k * M + m] = 0.F;
    }
  }
}

void hgemm_padding_A_Trans_wrt_K(const __fp16 *A, const __fp16 *Ap,
                                 unsigned int M, unsigned int K,
                                 unsigned int M8, unsigned int K8) {
  std::invalid_argument("NYI : hgemm_padding_A_Trans_wrt_K");
}

void hgemm_padding_A_Trans_wrt_MK(const __fp16 *A, const __fp16 *Ap,
                                  unsigned int M, unsigned int K,
                                  unsigned int M8, unsigned int K8) {
  std::invalid_argument("NYI : hgemm_padding_A_Trans_wrt_MK");
}
