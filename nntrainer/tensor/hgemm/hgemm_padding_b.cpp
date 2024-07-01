// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_padding_b.cpp
 * @date   05 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source file for padding function used in hgemm
 *
 */

#include <arm_neon.h>
#include <hgemm_padding_b.h>
#include <hgemm_util.h>
#include <iostream>
#include <stdexcept>

void hgemm_padding_B(const __fp16 *B, const __fp16 *Bp, unsigned int K,
                     unsigned int N, unsigned int K8, unsigned int N16,
                     bool transB) {
  if (transB) {
    hgemm_padding_B_Trans(B, Bp, K, N, K8, N16);
  } else {
    hgemm_padding_B_noTrans(B, Bp, K, N, K8, N16);
  }
}

void hgemm_padding_B_noTrans(const __fp16 *B, const __fp16 *Bp, unsigned int K,
                             unsigned int N, unsigned int K8,
                             unsigned int N16) {
  if (K != K8 && N != N16) {
    hgemm_padding_B_noTrans_wrt_KN(B, Bp, K, N, K8, N16);
  } else if (K != K8) {
    hgemm_padding_B_noTrans_wrt_K(B, Bp, K, N, K8, N16);
  } else if (N != N16) {
    hgemm_padding_B_noTrans_wrt_N(B, Bp, K, N, K8, N16);
  } else {
    std::invalid_argument("Error : No room for matrix B padding");
  }
}
void hgemm_padding_B_noTrans_wrt_N(const __fp16 *B, const __fp16 *Bp,
                                   unsigned int K, unsigned int N,
                                   unsigned int K8, unsigned int N16) {
  std::invalid_argument("NYI : hgemm_padding_B_noTrans_wrt_N");
}

void hgemm_padding_B_noTrans_wrt_K(const __fp16 *B, const __fp16 *Bp,
                                   unsigned int K, unsigned int N,
                                   unsigned int K8, unsigned int N16) {
  __fp16 *B8 = (__fp16 *)Bp;
  float16x8_t ZEROS = vmovq_n_f16(0.F);

  B8 = alignedMalloc(K8 * N);

  for (unsigned int k = 0; k < K; ++k) {
    for (unsigned int n = 0; n < N; n += 8) {
      vst1q_f16(&B8[k * N + n], vld1q_f16(&B[k * N + n]));
    }
  }
  for (unsigned int k = K; k < K8; ++k) {
    for (unsigned int n = 0; n < N; n += 8) {
      vst1q_f16(&B8[k * N + n], ZEROS);
    }
  }

/*MATRIX PRINT DEBUG*/
//   std::cout << "B Matrix\n";
//   for (unsigned int k = 0; k < K; ++k) {
//     for (unsigned int n = 0; n < N; n++) {
//       std::cout << B[k * N + n] << "\t";
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;

//   std::cout << "padded B Matrix\n";
//   for (unsigned int k = 0; k < K8_high; ++k) {
//     for (unsigned int n = 0; n < N; n++) {
//       std::cout << B8[k * N + n] << "\t";
//     }
//     std::cout << std::endl;
//   }
//   std::cout << std::endl;
}

void hgemm_padding_B_noTrans_wrt_KN(const __fp16 *B, const __fp16 *Bp,
                                    unsigned int K, unsigned int N,
                                    unsigned int K8, unsigned int N16) {
  std::invalid_argument("NYI : hgemm_padding_B_noTrans_wrt_KN");
}

void hgemm_padding_B_Trans(const __fp16 *B, const __fp16 *Bp, unsigned int K,
                           unsigned int N, unsigned int K8, unsigned int N16) {
  if (K != K8 && N != N16) {
    hgemm_padding_B_Trans_wrt_KN(B, Bp, K, N, K8, N16);
  } else if (K != K8) {
    hgemm_padding_B_Trans_wrt_K(B, Bp, K, N, K8, N16);
  } else if (N != N16) {
    hgemm_padding_B_Trans_wrt_N(B, Bp, K, N, K8, N16);
  } else {
    std::invalid_argument("Error : No room for matrix B padding");
  }
}

void hgemm_padding_B_Trans_wrt_N(const __fp16 *B, const __fp16 *Bp,
                                 unsigned int K, unsigned int N,
                                 unsigned int K8, unsigned int N16) {
  std::invalid_argument("NYI : hgemm_padding_B_Trans_wrt_N");
}

void hgemm_padding_B_Trans_wrt_K(const __fp16 *B, const __fp16 *Bp,
                                 unsigned int K, unsigned int N,
                                 unsigned int K8, unsigned int N16) {
  __fp16 *B8 = (__fp16 *)Bp;
  B8 = alignedMalloc(K8 * N);
  const unsigned int K8_low = (K >> 3) << 3;

  float16x8_t ZEROS = vmovq_n_f16(0.F);

  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K8_low; k += 8) {
      vst1q_f16(&B8[n * K8 + k], vld1q_f16(&B[n * K + k]));
    }
    for (unsigned int k = K8_low; k < K; ++k) {
      B8[n * K8 + k] = B[n * K + k];
    }
    for (unsigned int k = K; k < K8; ++k) {
      B8[n * K8 + k] = 0.F;
    }
  }

  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K; ++k) {
      if (B8[n * K8 + k] != B[n * K + k])
        std::cout << "FATAL B ERROR!\n";
    }
    for (unsigned int k = K; k < K8; ++k) {
      if (B8[n * K8 + k] != 0)
        std::cout << "FATAL non-zero B ERROR!\n";
    }
  }

  std::cout << "B Matrix\n";
  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = 0; k < K; ++k) {
      std::cout << B[n * K + k] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "padded B Matrix\n";
  for (unsigned int n = 0; n < N; ++n) {
    for (unsigned int k = K; k < K8; ++k) {
      std::cout << B8[n * K8 + k] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void hgemm_padding_B_Trans_wrt_KN(const __fp16 *B, const __fp16 *Bp,
                                  unsigned int K, unsigned int N,
                                  unsigned int K8, unsigned int N16) {
  std::invalid_argument("NYI : hgemm_padding_B_Trans_wrt_KN");
}
