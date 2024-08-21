// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_padding_b.h
 * @date   05 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a header file for padding function used in uhgemm
 *
 */

/**
 * @brief Padding function for matrix B in uhgemm
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 * @param transB Whether the matrix B is transposed or not
 */
void uhgemm_padding_B(const uint16_t *B, uint16_t *Bp, unsigned int K,
                     unsigned int N, unsigned int K8, unsigned int N16,
                     bool transB);

/**
 * @brief Padding function for non-transposed matrix B in uhgemm
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_noTrans(const uint16_t *B, uint16_t *Bp, unsigned int K,
                             unsigned int N, unsigned int K8, unsigned int N16);
/**
 * @brief Padding function for non-transposed matrix B in uhgemm w.r.t. N
 * direction
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_noTrans_wrt_N(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                   unsigned int N, unsigned int K8,
                                   unsigned int N16);
/**
 * @brief Padding function for non-transposed matrix B in uhgemm w.r.t. K
 * direction
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_noTrans_wrt_K(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                   unsigned int N, unsigned int K8,
                                   unsigned int N16);
/**
 * @brief Padding function for non-transposed matrix B in uhgemm w.r.t. N and K
 * direction
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_noTrans_wrt_KN(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                    unsigned int N, unsigned int K8,
                                    unsigned int N16);
/**
 * @brief Padding function for transposed matrix B in uhgemm
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_Trans(const uint16_t *B, uint16_t *Bp, unsigned int K,
                           unsigned int N, unsigned int K8, unsigned int N16);
/**
 * @brief Padding function for transposed matrix B in uhgemm w.r.t. N direction
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_Trans_wrt_N(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                 unsigned int N, unsigned int K8,
                                 unsigned int N16);
/**
 * @brief Padding function for transposed matrix B in uhgemm w.r.t. K direction
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_Trans_wrt_K(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                 unsigned int N, unsigned int K8,
                                 unsigned int N16);

/**
 * @brief Padding function for transposed matrix B in uhgemm w.r.t. K and N
 * direction
 *
 * @param B src matrix to pad
 * @param Bp dst matrix after padding
 * @param K the number of rows of matrix B
 * @param N the number of cols of matrix B
 * @param K8 Least multiple of 8 that is bigger than or equal to K
 * @param N16 Least multiple of 16 that is bigger than or equal to N
 */
void uhgemm_padding_B_Trans_wrt_KN(const uint16_t *B, uint16_t *Bp, unsigned int K,
                                  unsigned int N, unsigned int K8,
                                  unsigned int N16);
