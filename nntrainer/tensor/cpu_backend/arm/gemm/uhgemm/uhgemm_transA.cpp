// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_transA.cpp
 * @date   10 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is uint16 GEMM interface of transposed A case
 *
 */

#include <matrix_transpose_neon.h>
#include <uhgemm_noTrans.h>
#include <uhgemm_transA.h>
#include <gemm_util.h>

void uhgemm_transA(const uint16_t *A, const uint16_t *B, unsigned int *C,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int alpha, unsigned int beta) {
  uint16_t *A_T = alignedMalloc<uint16_t>(M * K);

  transpose_neon<uint16_t>(K, M, A, M, A_T, K);

  uhgemm_noTrans(A_T, B, C, M, N, K, alpha, beta);

  free(A_T);
}

void uhgemm_transA(const uint16_t *A, const uint16_t *B, uint16_t *C,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int alpha, unsigned int beta) {
  uint16_t *A_T = alignedMalloc<uint16_t>(M * K);

  transpose_neon<uint16_t>(K, M, A, M, A_T, K);

  uhgemm_noTrans(A_T, B, C, M, N, K, alpha, beta);

  free(A_T);
}
