// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   sqnbitgemm_interface.h
 * @date   29 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface for the nntrainer to use sqnbitgemm APIs
 *
 */
#include "mlas_q4.h"
#include "mlas_qnbit.h"

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_get_gqu4_rhs_nt_t_quant_size(size_t N, size_t K,
                                           size_t &QuantBDataSizeInBytes,
                                           size_t &QuantBScaleSize,
                                           size_t &QuantBZeroPointSizeInBytes);

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_gqu4_rhs_nt_t(const float *B, void *_QuantBData,
                            float *_QuantBScale, void *_QuantBZeroPoint,
                            size_t N, size_t K, bool Symmetric);

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_gqu4_gemm(size_t M, size_t N, size_t K, const float *A,
                        size_t lda, const void *QuantBData,
                        const float *QuantBScale, const void *QuantBZeroPoint,
                        const float *Bias, float *C, size_t ldc,
                        MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType = CompInt8,
                        // MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType = CompFp32,
                        MLAS_THREADPOOL *Threadpool = nullptr);

extern template void nntr_sqn_gqu4_rhs_nt_t<4, 64>(const float *, void *,
                                                   float *, void *, size_t,
                                                   size_t, bool);

extern template void
nntr_sqn_gqu4_gemm<4, 64>(size_t, size_t, size_t, const float *, size_t,
                          const void *, const float *, const void *,
                          const float *, float *, size_t,
                          MLAS_SQNBIT_GEMM_COMPUTE_TYPE, MLAS_THREADPOOL *);
extern template void
nntr_sqn_get_gqu4_rhs_nt_t_quant_size<4, 64>(size_t, size_t, size_t &, size_t &,
                                             size_t &);
