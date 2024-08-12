// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_noTrans.h
 * @date   10 July 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is half-precision GEMM interface of non-transposed case
 *
 */

#include <stdint.h>

/**
 * @brief uhgemm noTrans computation with 1x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_1x4(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, uint16_t *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 1x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_1x4(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, unsigned int *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 4x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_4x4(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, uint16_t *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 1x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_1x8(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, uint16_t *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 1x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_1x8(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, unsigned int *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 8x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_8x8(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, uint16_t *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 4x4 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_4x4(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, unsigned int *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 8x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_8x8(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, unsigned int *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 4x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_4x8(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, uint16_t *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 4x8 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_4x8(unsigned int M, unsigned int N, unsigned int K,
                       const uint16_t *A, unsigned int lda, const uint16_t *B,
                       unsigned int ldb, unsigned int *C, unsigned int ldc,
                       unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 8x16 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_8x16(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, uint16_t *C, unsigned int ldc,
                        unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief uhgemm noTrans computation with 8x16 kernel : C = A*B,
 *
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_8x16(unsigned int M, unsigned int N, unsigned int K,
                        const uint16_t *A, unsigned int lda, const uint16_t *B,
                        unsigned int ldb, unsigned int *C, unsigned int ldc,
                        unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief     uhgemm fallback with NEON : Y = alpha*op(A)*op(B) + beta*C,
 * @param M length of the row of matrix A
 * @param N length of the col of matrix B
 * @param K length of the col of matrix A
 * @param A input matrix A
 * @param lda length of the col of matrix A
 * @param B input matrix B
 * @param ldb length of the col of matrix B
 * @param C output matrix C
 * @param ldc length of the col of matrix C
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_fallback(unsigned int M, unsigned int N, unsigned int K,
                            const uint16_t *A, unsigned int lda, const uint16_t *B,
                            unsigned int ldb, unsigned int *C, unsigned int ldc,
                            unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief     uhgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans(const uint16_t *A, const uint16_t *B, unsigned int *C, unsigned int M,
                   unsigned int N, unsigned int K, unsigned int alpha = 1.F,
                   unsigned int beta = 0);
void uhgemm_noTrans(const uint16_t *A, const uint16_t *B, uint16_t *C, unsigned int M,
                   unsigned int N, unsigned int K, unsigned int alpha = 1.F,
                   unsigned int beta = 0);
/**
 * @brief     uhgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where M, N, K are divisible by at least 4
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C uint16_t * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_strict(const uint16_t *A, const uint16_t *B, uint16_t *C,
                          unsigned int M, unsigned int N, unsigned int K,
                          unsigned int alpha = 1.F, unsigned int beta = 0);

/**
 * @brief     uhgemm computation with neon : Y = alpha*op(A)*op(B) + beta*C,
 * where M, N, K are divisible by at least 4
 * @param[in] A uint16_t * for Matrix A
 * @param[in] B uint16_t * for Matrix B
 * @param[in] C unsigned int * for Matrix C
 * @param[in] M number of op(A)'s and C's row
 * @param[in] N number of op(B)'s and C's columns
 * @param[in] K number of op(A)'s and columns and op(B)'s rows
 * @param[in] alpha unsigned int number
 * @param[in] beta unsigned int number
 */
void uhgemm_noTrans_strict(const uint16_t *A, const uint16_t *B, unsigned int *C,
                          unsigned int M, unsigned int N, unsigned int K,
                          unsigned int alpha = 1.F, unsigned int beta = 0);
