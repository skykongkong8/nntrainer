// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface
 *
 */
#include <stdint.h>

/**
 * @brief     uhgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where op(X) is one of X or X**T
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C uint16_t * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void uhgemm(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta, bool TransA,
           bool TransB);
void uhgemm_pure(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
           unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta, bool TransA,
           bool TransB);
/**
 * @brief     uhgemm computation with neon but with small dim without padding : Y
 * = alpha*op(A)*op(B) + beta*C, where op(X) is one of X or X**T
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C uint16_t * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void uhgemm_small(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
                 unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta,
                 bool TransA, bool TransB);

/**
 * @brief     Checking function for whether matrix A or B needs padding for
 * optimal performance of fixed blocking-kernel sequence
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void uhgemm_ensure_divisibility(const uint16_t *A, const uint16_t *B, unsigned int *C32,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int alpha = 1.F, unsigned int beta = 0,
                               bool TransA = false, bool TransB = false);

void uhgemm_ensure_divisibility(const uint16_t *A, const uint16_t *B, uint16_t *C,
                               unsigned int M, unsigned int N, unsigned int K,
                               unsigned int alpha = 1.F, unsigned int beta = 0,
                               bool TransA = false, bool TransB = false);

/**
 * @brief     Classifying function for GEMM computation case for noTrans,
 * transA, transB, transAB
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C uint16_t * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void uhgemm_classify(const uint16_t *A, const uint16_t *B, unsigned int *C32,
                    unsigned int M, unsigned int N, unsigned int K,
                    unsigned int alpha = 1.F, unsigned int beta = 0, bool TransA = false,
                    bool TransB = false);
void uhgemm_classify(const uint16_t *A, const uint16_t *B, uint16_t *C,
                    unsigned int M, unsigned int N, unsigned int K,
                    unsigned int alpha = 1.F, unsigned int beta = 0, bool TransA = false,
                    bool TransB = false);
/**
 * @brief     uhgemm computation when K = 1. Transpose is mathematically no use
 * for here, and partial accumulation is also not needed.
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C uint16_t * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 * @param[in] TransA bool transpose info of lhs matrix
 * @param[in] TransB bool transpose info of rhs matrix
 */
void uhgemm_K1(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
              unsigned int N, unsigned int K, unsigned int alpha, unsigned int beta,
              bool TransA, bool TransB);
