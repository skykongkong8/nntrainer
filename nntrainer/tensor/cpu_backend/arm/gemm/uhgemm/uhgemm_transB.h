// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_transB.h
 * @date   10 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is uint16 GEMM interface of transposed B case
 *
 */

#include <stdint.h>

/**
 * @brief     uhgemm transB computation : Y = alpha*A*B_T + beta*C,
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transB(const uint16_t *A, const uint16_t *B, unsigned int *C,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int alpha, unsigned int beta);
/**
 * @brief     uhgemm transB computation : Y = alpha*A*B_T + beta*C,
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transB(const uint16_t *A, const uint16_t *B, uint16_t *C,
                   unsigned int M, unsigned int N, unsigned int K,
                   unsigned int alpha, unsigned int beta);
/**
 * @brief     uhgemm transB computation : Y = alpha*A*B_T + beta*C,
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transB_fallback(const uint16_t *A, const uint16_t *B,
                            unsigned int *C, unsigned int M, unsigned int N,
                            unsigned int K, unsigned int alpha,
                            unsigned int beta);
/**
 * @brief     uhgemm transB computation : Y = alpha*A*B_T + beta*C,
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transB_fallback(const uint16_t *A, const uint16_t *B, uint16_t *C,
                            unsigned int M, unsigned int N, unsigned int K,
                            unsigned int alpha, unsigned int beta);
/**
 * @brief     uhgemm transB computation with kernel 8x16
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transB_8x16(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, unsigned int *C, unsigned int ldc,
                        unsigned int alpha = 1.F, unsigned int beta = 0);
/**
 * @brief     uhgemm transB computation with kernel 8x16
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_transB_8x16(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, uint16_t *C, unsigned int ldc,
                        unsigned int alpha = 1.F, unsigned int beta = 0);
