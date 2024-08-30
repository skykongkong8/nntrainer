// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_transA.h
 * @date   10 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is uint16 GEMM interface of transposed A case
 *
 */

#include <stdint.h>

/**
 * @brief     uhgemm computation with neon : Y = alpha*A_T*B + beta*C,
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transA(const uint16_t *A, const uint16_t *B, unsigned int *C,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int alpha, unsigned int beta);

/**
 * @brief     uhgemm computation with neon : Y = alpha*A_T*B + beta*C,
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transA(const uint16_t *A, const uint16_t *B, uint16_t *C,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int alpha, unsigned int beta);
