// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_noTrans.cpp
 * @date   10 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is uint16 GEMM interface of non-transposed case
 *
 */

#include <arm_neon.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <matrix_transpose_neon.h>
#include <uhgemm_common.h>
#include <uhgemm_kernel.h>
#include <uhgemm_noTrans.h>
#include <uhgemm_pack.h>
#include <gemm_util.h>

void uhgemm_noTrans(const uint16_t *A, const uint16_t *B, unsigned int *C32,
                    unsigned int M, unsigned int N, unsigned int K,
                    unsigned int alpha, unsigned int beta) {
  if (alpha == 1) {
    uhgemm_noTrans_strict(A, B, C32, M, N, K, alpha, beta);
  } else {
    uhgemm_noTrans_fallback(M, N, K, A, K, B, N, C32, N, alpha, beta);
  }
}

void uhgemm_noTrans(const uint16_t *A, const uint16_t *B, uint16_t *C,
                    unsigned int M, unsigned int N, unsigned int K,
                    unsigned int alpha, unsigned int beta) {
  if (alpha == 1) {
    uhgemm_noTrans_strict(A, B, C, M, N, K, alpha, beta);
  } else {
    std::cerr
      << "[Error] uhgemm_noTrans without scale factor : should not reach!";
  }
}

void uhgemm_noTrans_strict(const uint16_t *A, const uint16_t *B,
                           unsigned int *C32, unsigned int M, unsigned int N,
                           unsigned int K, unsigned int alpha,
                           unsigned int beta) {
  // used bitwise operator instead of modulo for performance
  // e.g (M % 8) is same as (M & 0x7) which will extract last 3 bits of M
  if ((M & 0x7) == 0 && (N & 0xF) == 0 && (K & 0x7) == 0) {
    uhgemm_noTrans_8x16(M, N, K, A, K, B, N, C32, N, alpha, beta);
  } else if ((M & 0x7) == 0 && (N & 0x7) == 0 && (K & 0x7) == 0) {
    uhgemm_noTrans_8x8(M, N, K, A, K, B, N, C32, N, alpha, beta);
  } else if ((M & 0x3) == 0 && (N & 0x7) == 0 && (K & 0x7) == 0) {
    uhgemm_noTrans_4x8(M, N, K, A, K, B, N, C32, N, alpha, beta);
  } else if ((K & 0x7) == 0 && (N & 0x7) == 0) {
    uhgemm_noTrans_1x8(M, N, K, A, K, B, N, C32, N, alpha, beta);
  } else if ((K & 0x7) == 0 && (N & 0x3) == 0) {
    uhgemm_noTrans_1x4(M, N, K, A, K, B, N, C32, N, alpha, beta);
  } else {
    uhgemm_noTrans_fallback(M, N, K, A, K, B, N, C32, N, alpha, beta);
  }
}

void uhgemm_noTrans_strict(const uint16_t *A, const uint16_t *B, uint16_t *C,
                           unsigned int M, unsigned int N, unsigned int K,
                           unsigned int alpha, unsigned int beta) {
  if (alpha == 1.F) {
    // used bitwise operator instead of modulo for performance
    // e.g (M % 8) is same as (M & 0x7) which will extract last 3 bits of M
    if ((M & 0x7) == 0 && (N & 0xF) == 0 && (K & 0x7) == 0) {
      uhgemm_noTrans_8x16(M, N, K, A, K, B, N, C, N, alpha, beta);
    } else if ((M & 0x7) == 0 && (N & 0x7) == 0 && (K & 0x7) == 0) {
      uhgemm_noTrans_8x8(M, N, K, A, K, B, N, C, N, alpha, beta);
    } else if ((M & 0x3) == 0 && (N & 0x7) == 0 && (K & 0x3) == 0) {
      uhgemm_noTrans_4x8(M, N, K, A, K, B, N, C, N, alpha, beta);
    } else if ((M & 0x3) == 0 && (N & 0x3) == 0 && (K & 0x3) == 0) {
      uhgemm_noTrans_4x4(M, N, K, A, K, B, N, C, N, alpha, beta);
    } else if ((N & 0x7) == 0 && (K & 0x7) == 0) {
      uhgemm_noTrans_1x8(M, N, K, A, K, B, N, C, N, alpha, beta);
    } else if ((N & 0x3) == 0 && (K & 0x7) == 0) {
      uhgemm_noTrans_1x4(M, N, K, A, K, B, N, C, N, alpha, beta);
    }
  }
}

void uhgemm_noTrans_1x4(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, uint16_t *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }
      packing_B4(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_1) {
          m2_min = 3 * GEMM_UNROLLING_1;
        } else if (m2_min >= 2 * GEMM_UNROLLING_1) {
          m2_min = 2 * GEMM_UNROLLING_1;
        } else if (m2_min > GEMM_UNROLLING_1) {
          m2_min = GEMM_UNROLLING_1;
        }

        packing_A1(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms));

        uhgemm_kernel_1x4(m2_min, n_min, k_min, sa + k_min * (mms - ms), sb,
                          C + mms * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
        }

        packing_B4(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_1x4(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_1x4(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, unsigned int *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }
      packing_B4(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_1) {
          m2_min = 3 * GEMM_UNROLLING_1;
        } else if (m2_min >= 2 * GEMM_UNROLLING_1) {
          m2_min = 2 * GEMM_UNROLLING_1;
        } else if (m2_min > GEMM_UNROLLING_1) {
          m2_min = GEMM_UNROLLING_1;
        }

        packing_A1(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms));

        uhgemm_kernel_1x4(m2_min, n_min, k_min, sa + k_min * (mms - ms), sb,
                          C + mms * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
        }

        packing_B4(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_1x4(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_4x4(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, uint16_t *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }
      packing_B4(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_4) {
          m2_min = 3 * GEMM_UNROLLING_4;
        } else if (m2_min >= 2 * GEMM_UNROLLING_4) {
          m2_min = 2 * GEMM_UNROLLING_4;
        } else if (m2_min > GEMM_UNROLLING_4) {
          m2_min = GEMM_UNROLLING_4;
        }

        packing_A4(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms));

        uhgemm_kernel_4x4(m2_min, n_min, k_min, sa + k_min * (mms - ms), sb,
                          C + mms * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
        }

        packing_B4(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_4x4(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_1x8(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, uint16_t *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  unsigned int l1stride = 1;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = ((n_min / 2 + GEMM_UNROLLING_8 - 1) / GEMM_UNROLLING_8) *
                GEMM_UNROLLING_8;
      } else {
        l1stride = 0;
      }
      packing_B8(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_1) {
          m2_min = 3 * GEMM_UNROLLING_1;
        } else if (m2_min >= 2 * GEMM_UNROLLING_1) {
          m2_min = 2 * GEMM_UNROLLING_1;
        } else if (m2_min > GEMM_UNROLLING_1) {
          m2_min = GEMM_UNROLLING_1;
        }

        packing_A1(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms) * l1stride);

        uhgemm_kernel_1x8(m2_min, n_min, k_min,
                          sa + l1stride * k_min * (mms - ms), sb, C + mms * ldc,
                          ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
        }

        packing_B8(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_1x8(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_1x8(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, unsigned int *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  unsigned int l1stride = 1;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = ((n_min / 2 + GEMM_UNROLLING_8 - 1) / GEMM_UNROLLING_8) *
                GEMM_UNROLLING_8;
      } else {
        l1stride = 0;
      }
      packing_B8(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_1) {
          m2_min = 3 * GEMM_UNROLLING_1;
        } else if (m2_min >= 2 * GEMM_UNROLLING_1) {
          m2_min = 2 * GEMM_UNROLLING_1;
        } else if (m2_min > GEMM_UNROLLING_1) {
          m2_min = GEMM_UNROLLING_1;
        }

        packing_A1(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms) * l1stride);

        uhgemm_kernel_1x8(m2_min, n_min, k_min,
                          sa + l1stride * k_min * (mms - ms), sb, C + mms * ldc,
                          ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_1 - 1) & ~(GEMM_UNROLLING_1 - 1);
        }

        packing_B8(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_1x8(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_4x4(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, unsigned int *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }
      packing_B4(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_4) {
          m2_min = 3 * GEMM_UNROLLING_4;
        } else if (m2_min >= 2 * GEMM_UNROLLING_4) {
          m2_min = 2 * GEMM_UNROLLING_4;
        } else if (m2_min > GEMM_UNROLLING_4) {
          m2_min = GEMM_UNROLLING_4;
        }

        packing_A4(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms));

        uhgemm_kernel_4x4(m2_min, n_min, k_min, sa + k_min * (mms - ms), sb,
                          C + mms * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
        }

        packing_B4(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_4x4(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_4x8(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, uint16_t *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  unsigned int l1stride = 1;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = ((n_min / 2 + GEMM_UNROLLING_8 - 1) / GEMM_UNROLLING_8) *
                GEMM_UNROLLING_8;
      } else {
        l1stride = 0;
      }
      packing_B8(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_4) {
          m2_min = 3 * GEMM_UNROLLING_4;
        } else if (m2_min >= 2 * GEMM_UNROLLING_4) {
          m2_min = 2 * GEMM_UNROLLING_4;
        } else if (m2_min > GEMM_UNROLLING_4) {
          m2_min = GEMM_UNROLLING_4;
        }

        packing_A4(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms) * l1stride);

        uhgemm_kernel_4x8(m2_min, n_min, k_min,
                          sa + l1stride * k_min * (mms - ms), sb, C + mms * ldc,
                          ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
        }

        packing_B8(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_4x8(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_4x8(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, unsigned int *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {
  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  unsigned int l1stride = 1;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = ((n_min / 2 + GEMM_UNROLLING_8 - 1) / GEMM_UNROLLING_8) *
                GEMM_UNROLLING_8;
      } else {
        l1stride = 0;
      }
      packing_B8(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_4) {
          m2_min = 3 * GEMM_UNROLLING_4;
        } else if (m2_min >= 2 * GEMM_UNROLLING_4) {
          m2_min = 2 * GEMM_UNROLLING_4;
        } else if (m2_min > GEMM_UNROLLING_4) {
          m2_min = GEMM_UNROLLING_4;
        }

        packing_A4(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms) * l1stride);

        uhgemm_kernel_4x8(m2_min, n_min, k_min,
                          sa + l1stride * k_min * (mms - ms), sb, C + mms * ldc,
                          ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
        }

        packing_B8(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_4x8(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_8x8(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, uint16_t *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {

  uint16_t *sa = alignedMalloc<uint16_t>(M * K);
  uint16_t *sb = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, mms, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_8 - 1) & ~(GEMM_UNROLLING_8 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = (n_min / 2 + GEMM_UNROLLING_8 - 1) & ~(GEMM_UNROLLING_8 - 1);
      }
      packing_B8(k_min, n_min, B + ks * ldb, ldb, sb);

      for (mms = ms; mms < ms + m_min; mms += m2_min) {
        m2_min = (ms + m_min) - mms;
        if (m2_min >= 3 * GEMM_UNROLLING_8) {
          m2_min = 3 * GEMM_UNROLLING_8;
        } else if (m2_min >= 2 * GEMM_UNROLLING_8) {
          m2_min = 2 * GEMM_UNROLLING_8;
        } else if (m2_min > GEMM_UNROLLING_8) {
          m2_min = GEMM_UNROLLING_8;
        }

        packing_A8(m2_min, k_min, A + mms * lda + ks, lda,
                   sa + k_min * (mms - ms));

        uhgemm_kernel_8x8(m2_min, n_min, k_min, sa + k_min * (mms - ms), sb,
                          C + mms * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_8 - 1) & ~(GEMM_UNROLLING_8 - 1);
        }

        packing_B8(k_min, n_min, B + ns + ldb * ks, ldb, sb);
        uhgemm_kernel_8x8(m_min, n_min, k_min, sa, sb, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sa);
  free(sb);
}

void uhgemm_noTrans_8x8(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, unsigned int *C, unsigned int ldc,
                        unsigned int alpha, unsigned int beta) {

  uint16_t *sA = alignedMalloc<uint16_t>(M * K);
  uint16_t *sB = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, ms2, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_8 - 1) & ~(GEMM_UNROLLING_8 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = (n_min / 2 + GEMM_UNROLLING_8 - 1) & ~(GEMM_UNROLLING_8 - 1);
      }
      packing_B8(k_min, n_min, B + ks * ldb, ldb, sB);

      for (ms2 = ms; ms2 < ms + m_min; ms2 += m2_min) {
        m2_min = (ms + m_min) - ms2;
        if (m2_min >= 3 * GEMM_UNROLLING_8) {
          m2_min = 3 * GEMM_UNROLLING_8;
        } else if (m2_min >= 2 * GEMM_UNROLLING_8) {
          m2_min = 2 * GEMM_UNROLLING_8;
        } else if (m2_min > GEMM_UNROLLING_8) {
          m2_min = GEMM_UNROLLING_8;
        }

        packing_A8(m2_min, k_min, A + ms2 * lda + ks, lda,
                   sA + k_min * (ms2 - ms));

        uhgemm_kernel_8x8(m2_min, n_min, k_min, sA + k_min * (ms2 - ms), sB,
                          C + ms2 * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_8 - 1) & ~(GEMM_UNROLLING_8 - 1);
        }

        packing_B8(k_min, n_min, B + ns + ldb * ks, ldb, sB);
        uhgemm_kernel_8x8(m_min, n_min, k_min, sA, sB, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sA);
  free(sB);
}

void uhgemm_noTrans_8x16(unsigned int M, unsigned int N, unsigned int K,
                         const uint16_t *A, unsigned int lda, const uint16_t *B,
                         unsigned int ldb, uint16_t *C, unsigned int ldc,
                         unsigned int alpha, unsigned int beta) {
  uint16_t *sA = alignedMalloc<uint16_t>(M * K);
  uint16_t *sB = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, ms2, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  unsigned int stride_l1 = 1;

  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = ((n_min / 2 + GEMM_UNROLLING_16 - 1) / GEMM_UNROLLING_16) *
                GEMM_UNROLLING_16;
      } else {
        stride_l1 = 0;
      }
      packing_B16(k_min, n_min, B + ks * ldb, ldb, sB);

      for (ms2 = ms; ms2 < ms + m_min; ms2 += m2_min) {
        m2_min = (ms + m_min) - ms2;
        if (m2_min >= 3 * GEMM_UNROLLING_8) {
          m2_min = 3 * GEMM_UNROLLING_8;
        } else if (m2_min >= 2 * GEMM_UNROLLING_8) {
          m2_min = 2 * GEMM_UNROLLING_8;
        } else if (m2_min > GEMM_UNROLLING_8) {
          m2_min = GEMM_UNROLLING_8;
        }

        packing_A8(m2_min, k_min, A + ms2 * lda + ks, lda,
                   sA + k_min * (ms2 - ms) * stride_l1);

        uhgemm_kernel_8x16(m2_min, n_min, k_min,
                           sA + stride_l1 * k_min * (ms2 - ms), sB,
                           C + ms2 * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min = (n_min / 2 + GEMM_UNROLLING_8 - 1) & ~(GEMM_UNROLLING_8 - 1);
        }

        packing_B16(k_min, n_min, B + ns + ldb * ks, ldb, sB);
        uhgemm_kernel_8x16(m_min, n_min, k_min, sA, sB, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sA);
  free(sB);
}

void uhgemm_noTrans_8x16(unsigned int M, unsigned int N, unsigned int K,
                         const uint16_t *A, unsigned int lda, const uint16_t *B,
                         unsigned int ldb, unsigned int *C, unsigned int ldc,
                         unsigned int alpha, unsigned int beta) {
  uint16_t *sA = alignedMalloc<uint16_t>(M * K);
  uint16_t *sB = alignedMalloc<uint16_t>(K * N);

  unsigned int ms, ms2, ns, ks;
  unsigned int m_min, m2_min, n_min, k_min;
  unsigned int stride_l1 = 1;

  for (ms = 0; ms < M; ms += M_BLOCKING) {
    m_min = M - ms;
    if (m_min > M_BLOCKING) {
      m_min = M_BLOCKING;
    }

    for (ks = 0; ks < K; ks += k_min) {
      k_min = K - ks;
      if (k_min >= (K_BLOCKING << 1)) {
        k_min = K_BLOCKING;
      } else if (k_min > K_BLOCKING) {
        k_min = (k_min / 2 + GEMM_UNROLLING_4 - 1) & ~(GEMM_UNROLLING_4 - 1);
      }

      n_min = N;
      if (N >= N_BLOCKING * 2) {
        n_min = N_BLOCKING;
      } else if (N > N_BLOCKING) {
        n_min = ((n_min / 2 + GEMM_UNROLLING_16 - 1) / GEMM_UNROLLING_16) *
                GEMM_UNROLLING_16;
      } else {
        stride_l1 = 0;
      }
      packing_B16(k_min, n_min, B + ks * ldb, ldb, sB);

      for (ms2 = ms; ms2 < ms + m_min; ms2 += m2_min) {
        m2_min = (ms + m_min) - ms2;
        if (m2_min >= 3 * GEMM_UNROLLING_8) {
          m2_min = 3 * GEMM_UNROLLING_8;
        } else if (m2_min >= 2 * GEMM_UNROLLING_8) {
          m2_min = 2 * GEMM_UNROLLING_8;
        } else if (m2_min > GEMM_UNROLLING_8) {
          m2_min = GEMM_UNROLLING_8;
        }

        packing_A8(m2_min, k_min, A + ms2 * lda + ks, lda,
                   sA + k_min * (ms2 - ms));

        uhgemm_kernel_8x16(m2_min, n_min, k_min, sA + k_min * (ms2 - ms), sB,
                           C + ms2 * ldc, ldc);
      }

      for (ns = n_min; ns < N; ns += n_min) {
        n_min = N - ns;
        if (n_min >= N_BLOCKING * 2) {
          n_min = N_BLOCKING;
        } else if (n_min > N_BLOCKING) {
          n_min =
            (n_min / 2 + GEMM_UNROLLING_16 - 1) & ~(GEMM_UNROLLING_16 - 1);
        }

        packing_B16(k_min, n_min, B + ns + ldb * ks, ldb, sB);

        uhgemm_kernel_8x16(m_min, n_min, k_min, sA, sB, C + ms * ldc + ns, ldc);
      }
    }
  }

  free(sA);
  free(sB);
}

void uhgemm_noTrans_fallback(unsigned int M, unsigned int N, unsigned int K,
                             const uint16_t *A, unsigned int lda,
                             const uint16_t *B, unsigned int ldb,
                             unsigned int *C, unsigned int ldc,
                             unsigned int alpha, unsigned int beta) {

  unsigned int k = 0;
  unsigned int N8 = (N >> 3) << 3;
  uint16_t a[16];
  for (; (K - k) >= 16; k += 16) {
    for (unsigned int m = 0; m < M; m++) {
      vst1q_u16(&a[0], vmulq_n_u16(vld1q_u16(&A[m * K + k]), alpha));
      vst1q_u16(&a[8], vmulq_n_u16(vld1q_u16(&A[m * K + k + 8]), alpha));
      for (unsigned int n = 0; n < N8; n += 8) {
        uint16x8_t b0_7_0 = vmulq_n_u16(vld1q_u16(&B[k * N + n]), a[0]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 1) * N + n]), a[1]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 2) * N + n]), a[2]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 3) * N + n]), a[3]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 4) * N + n]), a[4]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 5) * N + n]), a[5]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 6) * N + n]), a[6]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 7) * N + n]), a[7]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 8) * N + n]), a[8]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 9) * N + n]), a[9]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 10) * N + n]), a[10]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 11) * N + n]), a[11]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 12) * N + n]), a[12]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 13) * N + n]), a[13]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 14) * N + n]), a[14]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 15) * N + n]), a[15]);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(&C[m * N + n]), vmovl_u16(vget_low_u16(b0_7_0)));
        uint32x4_t c0_7_high_32 = vaddq_u32(vld1q_u32(&C[m * N + n + 4]),
                                            vmovl_u16(vget_high_u16(b0_7_0)));

        vst1q_u32(&C[m * N + n], c0_7_low_32);
        vst1q_u32(&C[m * N + n + 4], c0_7_high_32);
      }
      if (N != N8) {
        unsigned int n = N8;
        uint16_t valsB_0[8];
        uint16_t valsB_1[8];
        uint16_t valsB_2[8];
        uint16_t valsB_3[8];
        uint16_t valsB_4[8];
        uint16_t valsB_5[8];
        uint16_t valsB_6[8];
        uint16_t valsB_7[8];
        uint16_t valsB_8[8];
        uint16_t valsB_9[8];
        uint16_t valsB_10[8];
        uint16_t valsB_11[8];
        uint16_t valsB_12[8];
        uint16_t valsB_13[8];
        uint16_t valsB_14[8];
        uint16_t valsB_15[8];
        unsigned int valsC[8];
        for (unsigned int idx = n; idx < N; idx++) {
          valsB_0[idx - n] = B[k * N + idx];
          valsB_1[idx - n] = B[(k + 1) * N + idx];
          valsB_2[idx - n] = B[(k + 2) * N + idx];
          valsB_3[idx - n] = B[(k + 3) * N + idx];
          valsB_4[idx - n] = B[(k + 4) * N + idx];
          valsB_5[idx - n] = B[(k + 5) * N + idx];
          valsB_6[idx - n] = B[(k + 6) * N + idx];
          valsB_7[idx - n] = B[(k + 7) * N + idx];
          valsB_8[idx - n] = B[(k + 8) * N + idx];
          valsB_9[idx - n] = B[(k + 9) * N + idx];
          valsB_10[idx - n] = B[(k + 10) * N + idx];
          valsB_11[idx - n] = B[(k + 11) * N + idx];
          valsB_12[idx - n] = B[(k + 12) * N + idx];
          valsB_13[idx - n] = B[(k + 13) * N + idx];
          valsB_14[idx - n] = B[(k + 14) * N + idx];
          valsB_15[idx - n] = B[(k + 15) * N + idx];
          valsC[idx - n] = C[m * N + idx];
        }

        uint16x8_t b = vmulq_n_u16(vld1q_u16(valsB_0), a[0]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_1), a[1]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_2), a[2]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_3), a[3]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_4), a[4]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_5), a[5]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_6), a[6]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_7), a[7]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_8), a[8]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_9), a[9]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_10), a[10]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_11), a[11]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_12), a[12]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_13), a[13]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_14), a[14]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_15), a[15]);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(valsC), vmovl_u16(vget_low_u16(b)));

        uint32x4_t c0_7_high_32 =
          vaddq_u32(vld1q_u32(valsC + 4), vmovl_u16(vget_high_u16(b)));

        vst1q_u32(valsC, c0_7_low_32);
        vst1q_u32(valsC + 4, c0_7_high_32);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }

  for (; (K - k) >= 8; k += 8) {
    for (unsigned int m = 0; m < M; m++) {
      vst1q_u16(a, vmulq_n_u16(vld1q_u16(&A[m * K + k]), alpha));

      for (unsigned int n = 0; n < N8; n += 8) {
        uint16x8_t b0_7_0 = vmulq_n_u16(vld1q_u16(&B[k * N + n]), a[0]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 1) * N + n]), a[1]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 2) * N + n]), a[2]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 3) * N + n]), a[3]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 4) * N + n]), a[4]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 5) * N + n]), a[5]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 6) * N + n]), a[6]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 7) * N + n]), a[7]);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(&C[m * N + n]), vmovl_u16(vget_low_u16(b0_7_0)));
        uint32x4_t c0_7_high_32 = vaddq_u32(vld1q_u32(&C[m * N + n + 4]),
                                            vmovl_u16(vget_high_u16(b0_7_0)));

        vst1q_u32(&C[m * N + n], c0_7_low_32);
        vst1q_u32(&C[m * N + n + 4], c0_7_high_32);
      }
      if (N != N8) {
        unsigned int n = N8;
        uint16_t valsB_0[8];
        uint16_t valsB_1[8];
        uint16_t valsB_2[8];
        uint16_t valsB_3[8];
        uint16_t valsB_4[8];
        uint16_t valsB_5[8];
        uint16_t valsB_6[8];
        uint16_t valsB_7[8];
        unsigned int valsC[8];
        for (unsigned int idx = n; idx < N; idx++) {
          valsB_0[idx - n] = B[k * N + idx];
          valsB_1[idx - n] = B[(k + 1) * N + idx];
          valsB_2[idx - n] = B[(k + 2) * N + idx];
          valsB_3[idx - n] = B[(k + 3) * N + idx];
          valsB_4[idx - n] = B[(k + 4) * N + idx];
          valsB_5[idx - n] = B[(k + 5) * N + idx];
          valsB_6[idx - n] = B[(k + 6) * N + idx];
          valsB_7[idx - n] = B[(k + 7) * N + idx];
          valsC[idx - n] = C[m * N + idx];
        }

        uint16x8_t b = vmulq_n_u16(vld1q_u16(valsB_0), a[0]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_1), a[1]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_2), a[2]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_3), a[3]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_4), a[4]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_5), a[5]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_6), a[6]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_7), a[7]);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(valsC), vmovl_u16(vget_low_u16(b)));

        uint32x4_t c0_7_high_32 =
          vaddq_u32(vld1q_u32(valsC + 4), vmovl_u16(vget_high_u16(b)));

        vst1q_u32(valsC, c0_7_low_32);
        vst1q_u32(valsC + 4, c0_7_high_32);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }

  for (; (K - k) >= 4; k += 4) {
    for (unsigned int m = 0; m < M; m++) {
      vst1_u16(a, vmul_n_u16(vld1_u16(&A[m * K + k]), alpha));

      for (unsigned int n = 0; n < N8; n += 8) {

        uint16x8_t b0_7_0 = vmulq_n_u16(vld1q_u16(&B[k * N + n]), a[0]);
        b0_7_0 = vmlaq_n_u16(b0_7_0, vld1q_u16(&B[(k + 1) * N + n]), a[1]);
        uint16x8_t b0_7_2 = vmulq_n_u16(vld1q_u16(&B[(k + 2) * N + n]), a[2]);
        b0_7_2 = vmlaq_n_u16(b0_7_2, vld1q_u16(&B[(k + 3) * N + n]), a[3]);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(&C[m * N + n]), vmovl_u16(vget_low_u16(b0_7_0)));
        uint32x4_t c0_7_high_32 = vaddq_u32(vld1q_u32(&C[m * N + n + 4]),
                                            vmovl_u16(vget_high_u16(b0_7_0)));

        c0_7_low_32 = vaddq_u32(c0_7_low_32, vmovl_u16(vget_low_u16(b0_7_2)));
        c0_7_high_32 =
          vaddq_u32(c0_7_high_32, vmovl_u16(vget_high_u16(b0_7_2)));

        vst1q_u32(&C[m * N + n], c0_7_low_32);
        vst1q_u32(&C[m * N + n + 4], c0_7_high_32);
      }
      if (N != N8) {
        unsigned int n = N8;
        uint16_t valsB_0[8];
        uint16_t valsB_1[8];
        uint16_t valsB_2[8];
        uint16_t valsB_3[8];
        unsigned int valsC[8];
        for (unsigned int idx = n; idx < N; idx++) {
          valsB_0[idx - n] = B[k * N + idx];
          valsB_1[idx - n] = B[(k + 1) * N + idx];
          valsB_2[idx - n] = B[(k + 2) * N + idx];
          valsB_3[idx - n] = B[(k + 3) * N + idx];
          valsC[idx - n] = C[m * N + idx];
        }

        uint16x8_t b = vmulq_n_u16(vld1q_u16(valsB_0), a[0]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_1), a[1]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_2), a[2]);
        b = vmlaq_n_u16(b, vld1q_u16(valsB_3), a[3]);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(valsC), vmovl_u16(vget_low_u16(b)));

        uint32x4_t c0_7_high_32 =
          vaddq_u32(vld1q_u32(valsC + 4), vmovl_u16(vget_high_u16(b)));

        vst1q_u32(valsC, c0_7_low_32);
        vst1q_u32(valsC + 4, c0_7_high_32);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }

  for (; k < K; k++) {
    for (unsigned int m = 0; m < M; m++) {
      uint16_t a0 = alpha * A[m * K + k];

      for (unsigned int n = 0; n < N8; n += 8) {
        uint16x8_t b0_7 = vmulq_n_u16(vld1q_u16(&B[k * N + n]), a0);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(&C[m * N + n]), vmovl_u16(vget_low_u16(b0_7)));

        uint32x4_t c0_7_high_32 = vaddq_u32(vld1q_u32(&C[m * N + n + 4]),
                                            vmovl_u16(vget_high_u16(b0_7)));

        vst1q_u32(&C[m * N + n], c0_7_low_32);
        vst1q_u32(&C[m * N + n + 4], c0_7_high_32);
      }
      if (N != N8) {
        unsigned int n = N8;
        uint16_t valsB[8];
        unsigned int valsC[8];
        for (unsigned int idx = n; idx < N; idx++) {
          valsB[idx - n] = B[k * N + idx];
          valsC[idx - n] = C[m * N + idx];
        }

        uint16x8_t b = vmulq_n_u16(vld1q_u16(valsB), a0);

        uint32x4_t c0_7_low_32 =
          vaddq_u32(vld1q_u32(valsC), vmovl_u16(vget_low_u16(b)));

        uint32x4_t c0_7_high_32 =
          vaddq_u32(vld1q_u32(valsC + 4), vmovl_u16(vget_high_u16(b)));

        vst1q_u32(valsC, c0_7_low_32);
        vst1q_u32(valsC + 4, c0_7_high_32);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }
}
