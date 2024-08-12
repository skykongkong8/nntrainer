// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm.cpp
 * @date   03 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface
 *
 */

#include <arm_neon.h>
#include <cmath>
#include <uhgemm.h>
#include <uhgemm_common.h>
#include <uhgemm_noTrans.h>
#include <uhgemm_padding.h>
#include <uhgemm_transA.h>
#include <uhgemm_transAB.h>
#include <uhgemm_transB.h>
#include <uhgemm_util.h>
#include <limits>
#include <iostream>

void uhgemm(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta, bool TransA,
           bool TransB) {
  if (K == 1) {
    return uhgemm_K1(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  } else if (M < 8 && K < 16 && N < 16) {
    return uhgemm_small(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  }

  const unsigned int M8_high = get_next_mltpl_of_n(M, 8);
  const unsigned int K8_high = get_next_mltpl_of_n(K, 8);
  const unsigned int N16_high = get_next_mltpl_of_n(N, 16);
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);

  uint32x4_t ZEROS = vmovq_n_u32(0);

  unsigned int *C32 = (unsigned int *)malloc(M8_high * N16_high * sizeof(unsigned int));

  unsigned int size = M8_high * N16_high;
  unsigned int size8 = get_prev_mltpl_of_2p_n(size, 3);
  unsigned int size4 = get_prev_mltpl_of_2p_n(size, 2);

  if (beta != 0) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N8_low; n += 8) {
        uint16x8_t c =
          vmulq_n_u16(vld1q_u16(&C[m * N + n]), static_cast<uint16_t>(beta));
        vst1q_u32(&C32[m * N16_high + n], vmovl_u16(vget_low_u16(c)));
        vst1q_u32(&C32[m * N16_high + n + 4], vmovl_u16(vget_high_u16(c)));
      }
      for (unsigned int n = N8_low; n < N; ++n) {
        C32[m * N16_high + n] = beta * C[m * N + n];
      }
      for (unsigned int n = N; n < N16_high; ++n) {
        C32[m * N16_high + n] = 0;
      }
    }
    for (unsigned m = M; m < M8_high; ++m) {
      for (unsigned int n = 0; n < N16_high; n += 4) {
        vst1q_u32(&C32[m * N16_high + n], ZEROS);
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

  uhgemm_ensure_divisibility(A, B, C32, M, N, K, alpha, beta, TransA, TransB);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      uint32x4_t x1 = vld1q_u32(&C32[m * N16_high + n]);
      uint32x4_t x2 = vld1q_u32(&C32[m * N16_high + n + 4]);
      vst1q_u16(&C[m * N + n],
                vcombine_u16(vmovn_u32(x1), vmovn_u32(x2)));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      C[m * N + n] = C32[m * N16_high + n];
    }
  }

  free(C32);
}

void uhgemm_pure(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta, bool TransA,
           bool TransB) {
  if (K == 1) {
    return uhgemm_K1(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  } else if (M < 8 && K < 16 && N < 16) {
    return uhgemm_small(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  }

  const unsigned int M8_high = get_next_mltpl_of_n(M, 8);
  const unsigned int K8_high = get_next_mltpl_of_n(K, 8);
  const unsigned int N16_high = get_next_mltpl_of_n(N, 16);
  const unsigned int N8_low = get_prev_mltpl_of_2p_n(N, 3);

  uint16x8_t ZEROS = vmovq_n_u16(0);

  uint16_t *C_tmp = (uint16_t *)malloc(M8_high * N16_high * sizeof(uint16_t));

  unsigned int size = M8_high * N16_high;
  unsigned int size8 = get_prev_mltpl_of_2p_n(size, 3);
  unsigned int size4 = get_prev_mltpl_of_2p_n(size, 2);

  if (beta != 0) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N8_low; n += 8) {
        vst1q_u16(&C_tmp[m * N16_high + n], vmulq_n_u16(vld1q_u16(&C[m * N + n]), static_cast<uint16_t>(beta)));
      }
      for (unsigned int n = N8_low; n < N; ++n) {
        C_tmp[m * N16_high + n] = beta * C[m * N + n];
      }
      for (unsigned int n = N; n < N16_high; ++n) {
        C_tmp[m * N16_high + n] = 0;
      }
    }
    for (unsigned m = M; m < M8_high; ++m) {
      for (unsigned int n = 0; n < N16_high; n += 8) {
        vst1q_u16(&C_tmp[m * N16_high + n], ZEROS);
      }
    }
  } else {
    for (unsigned int idx = 0; idx < size4; idx += 4) {
      vst1q_u16(&C_tmp[idx], ZEROS);
    }
    for (unsigned int idx = size4; idx < size; idx++) {
      C_tmp[idx] = 0;
    }
  }

  uhgemm_ensure_divisibility(A, B, C_tmp, M, N, K, alpha, beta, TransA, TransB);

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N8_low; n += 8) {
      vst1q_u16(&C[m * N + n], vld1q_u16(&C_tmp[m * N16_high + n]));
    }
    for (unsigned int n = N8_low; n < N; ++n) {
      C[m * N + n] = C_tmp[m * N16_high + n];
    }
  }

  free(C_tmp);
}

void uhgemm_small(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
                 unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta,
                 bool TransA, bool TransB) {
  unsigned int *C32 = (unsigned int *)malloc(M * N * sizeof(float));

  copy_C_to_C32(C, C32, M, N, beta);

  uhgemm_classify(A, B, C32, M, N, K, alpha, beta, TransA, TransB);

  copy_C32_to_C(C32, C, M, N, beta);

  free(C32);
}

void uhgemm_ensure_divisibility(const uint16_t *A, const uint16_t *B, unsigned int *C32,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int alpha, unsigned int beta, bool TransA,
                               bool TransB) {
  /// @note Padding standard : 8x16 is the only KERNEL that outperforms single
  /// precision GEMM 'so far'. Padding will forcibly make every GEMM cases to
  /// use it. Note that padding is not an optimal way here, but just an option
  /// that is easier to implement. Fine-grained packing, blocking, and
  /// corresponding kernels should be supported in the future for optimal
  /// performance in terms of both latency and memory.

  uint16_t *A_ = (uint16_t *)A, *B_ = (uint16_t *)B;
  unsigned int M_ = M, N_ = N, K_ = K;
  bool pad_A = false, pad_B = false;

  uint16_t *Ap;
  uint16_t *Bp;

  const unsigned int M8_high = ((M - 1) / 8 + 1) * 8;
  const unsigned int K8_high = ((K - 1) / 16 + 1) * 16;
  const unsigned int N16_high = ((N - 1) / 16 + 1) * 16;

  if ((M8_high != M) || (K8_high != K)) {
    pad_A = true;
    Ap = alignedMalloc(M8_high * K8_high);
    uhgemm_padding_A(A, Ap, M, K, M8_high, K8_high, TransA);
    A_ = Ap;
    M_ = M8_high;
    K_ = K8_high;
  }
  if ((K8_high != K) || (N16_high != N)) {
    pad_B = true;
    Bp = alignedMalloc(K8_high * N16_high);
    uhgemm_padding_B(B, Bp, K, N, K8_high, N16_high, TransB);
    B_ = Bp;
    K_ = K8_high;
    N_ = N16_high;
  }

  // std::cerr << "uhgemm_ensure_divisibility\n";
  // for (unsigned int i = 0; i < 5; ++i) {
  //   std::cerr << A[i] << "\t";
  // }
  // std::cerr << "\n";
  uhgemm_classify(A_, B_, C32, M_, N_, K_, alpha, beta, TransA, TransB);

  if (pad_A)
    free(Ap);
  if (pad_B)
    free(Bp);
}

void uhgemm_ensure_divisibility(const uint16_t *A, const uint16_t *B, uint16_t *C,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int alpha, unsigned int beta, bool TransA,
                               bool TransB) {
  /// @note Padding standard : 8x16 is the only KERNEL that outperforms single
  /// precision GEMM 'so far'. Padding will forcibly make every GEMM cases to
  /// use it. Note that padding is not an optimal way here, but just an option
  /// that is easier to implement. Fine-grained packing, blocking, and
  /// corresponding kernels should be supported in the future for optimal
  /// performance in terms of both latency and memory.

  uint16_t *A_ = (uint16_t *)A, *B_ = (uint16_t *)B;
  unsigned int M_ = M, N_ = N, K_ = K;
  bool pad_A = false, pad_B = false;

  uint16_t *Ap;
  uint16_t *Bp;

  const unsigned int M8_high = ((M - 1) / 8 + 1) * 8;
  const unsigned int K8_high = ((K - 1) / 16 + 1) * 16;
  const unsigned int N16_high = ((N - 1) / 16 + 1) * 16;

  if ((M8_high != M) || (K8_high != K)) {
    pad_A = true;
    Ap = alignedMalloc(M8_high * K8_high);
    uhgemm_padding_A(A, Ap, M, K, M8_high, K8_high, TransA);
    A_ = Ap;
    M_ = M8_high;
    K_ = K8_high;
  }
  if ((K8_high != K) || (N16_high != N)) {
    pad_B = true;
    Bp = alignedMalloc(K8_high * N16_high);
    uhgemm_padding_B(B, Bp, K, N, K8_high, N16_high, TransB);
    B_ = Bp;
    K_ = K8_high;
    N_ = N16_high;
  }

  uhgemm_classify(A_, B_, C, M_, N_, K_, alpha, beta, TransA, TransB);

  if (pad_A)
    free(Ap);
  if (pad_B)
    free(Bp);
}

void uhgemm_classify(const uint16_t *A, const uint16_t *B, unsigned int *C32,
                    unsigned int M, unsigned int N, unsigned int K, unsigned int alpha,
                    unsigned int beta, bool TransA, bool TransB) {
  if (!TransA && !TransB) {
    uhgemm_noTrans(A, B, C32, M, N, K, alpha, beta);
  } else if (TransA && !TransB) {
    uhgemm_transA(A, B, C32, M, N, K, alpha, beta);
  } else if (!TransA && TransB) {
    uhgemm_transB(A, B, C32, M, N, K, alpha, beta);
  } else { // TransA && TransB
    uhgemm_transAB(A, B, C32, M, N, K, alpha, beta);
  }
}

void uhgemm_classify(const uint16_t *A, const uint16_t *B, uint16_t *C32,
                    unsigned int M, unsigned int N, unsigned int K, unsigned int alpha,
                    unsigned int beta, bool TransA, bool TransB) {
  if (!TransA && !TransB) {
    uhgemm_noTrans(A, B, C32, M, N, K, alpha, beta);
  } else if (TransA && !TransB) {
    uhgemm_transA(A, B, C32, M, N, K, alpha, beta);
  } else if (!TransA && TransB) {
    uhgemm_transB(A, B, C32, M, N, K, alpha, beta);
  } else { // TransA && TransB
    uhgemm_transAB(A, B, C32, M, N, K, alpha, beta);
  }
}

void uhgemm_K1(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
              unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta,
              bool TransA, bool TransB) {
  unsigned int lda = (TransA) ? M : K;
  unsigned int ldb = (TransB) ? K : N;
  unsigned int ldc = N;

  
  uint16x8_t a_vec;
  unsigned int N8 = (N >> 3) << 3;
  for (unsigned int m = 0; m < M; ++m) {
    a_vec = vmovq_n_u16(alpha * A[m]);
    if (beta != 0) {
      for (unsigned int n = 0; n < N8; n += 8) {
        vst1q_u16(&C[m * ldc + n],
                  vaddq_u16(vmulq_u16(a_vec, vld1q_u16(&B[n])),
                            vmulq_n_u16(vld1q_u16(&C[m * ldc + n]), beta)));
      }
    } else {
      for (unsigned int n = 0; n < N8; n += 8) {
        vst1q_u16(&C[m * ldc + n], vmulq_u16(a_vec, vld1q_u16(&B[n])));
      }
    }
    for (unsigned int n = N8; n < N; ++n) {
      C[m * ldc + n] = alpha * A[m] * B[n] + beta * C[m * ldc + n];
    }
  }
}
