// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_transA.cpp
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of transposed A case
 *
 */

#include <uhgemm_noTrans.h>
#include <uhgemm_transA.h>
#include <uhgemm_util.h>
#include <matrix_transpose_neon.h>

void uhgemm_transA(const uint16_t *A, const uint16_t *B, unsigned int *C, unsigned int M,
                  unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta) {
  uint16_t *A_T = alignedMalloc(M * K);

  transpose_neon<uint16_t>(K, M, A, M, A_T, K);

  uhgemm_noTrans(A_T, B, C, M, N, K, alpha, beta);

  free(A_T);
}
