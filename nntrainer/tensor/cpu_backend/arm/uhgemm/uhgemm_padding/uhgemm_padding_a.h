// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_padding_a.h
 * @date   05 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a header file for padding function used in uhgemm
 *
 */

/**
 * @brief Padding function for matrix A in uhgemm
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param transA Whether the matrix A is transposed or not
 */
void uhgemm_padding_A(const uint16_t *A, uint16_t *Ap, unsigned int M,
                     unsigned int K, unsigned int M8, unsigned int K8,
                     bool transA);

/**
 * @brief Padding function for non-transposed matrix A in uhgemm
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_noTrans(const uint16_t *A, uint16_t *Ap, unsigned int M,
                             unsigned int K, unsigned int M8, unsigned int K8);

/**
 * @brief Padding function for non-transposed matrix A in uhgemm w.r.t. M
 * direction
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_noTrans_wrt_M(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                   unsigned int K, unsigned int M8,
                                   unsigned int K8);
/**
 * @brief Padding function for non-transposed matrix A in uhgemm w.r.t. K
 * direction
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_noTrans_wrt_K(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                   unsigned int K, unsigned int M8,
                                   unsigned int K8);

/**
 * @brief Padding function for non-transposed matrix A in uhgemm w.r.t. M and K
 * direction
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_noTrans_wrt_MK(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                    unsigned int K, unsigned int M8,
                                    unsigned int K8);
/**
 * @brief Padding function for transposed matrix A in uhgemm
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_Trans(const uint16_t *A, uint16_t *Ap, unsigned int M,
                           unsigned int K, unsigned int M8, unsigned int K8);
/**
 * @brief Padding function for transposed matrix A in uhgemm w.r.t. M direction
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_Trans_wrt_M(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                 unsigned int K, unsigned int M8,
                                 unsigned int K8);
/**
 * @brief Padding function for transposed matrix A in uhgemm w.r.t. K direction
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_Trans_wrt_K(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                 unsigned int K, unsigned int M8,
                                 unsigned int K8);
/**
 * @brief Padding function for transposed matrix A in uhgemm w.r.t. M and K
 * direction
 *
 * @param A src matrix to pad
 * @param Ap dst matrix after padding
 * @param M the number of rows of matrix A
 * @param K the number of cols of matrix A
 * @param M8 Least multiple of 8 that is bigger than or equal to M
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 */
void uhgemm_padding_A_Trans_wrt_MK(const uint16_t *A, uint16_t *Ap, unsigned int M,
                                  unsigned int K, unsigned int M8,
                                  unsigned int K8);
