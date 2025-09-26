// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   sqnbitgemm_interface.cpp
 * @date   29 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Function interface for the nntrainer to use sqnbitgemm APIs
 *
 */

#include <iostream>
#include <sqnbitgemm_interface.h>
#include <test_util.h>

template <size_t BlkBitWidth, size_t BlkLen>
void CallGemm(size_t M, size_t N, size_t K, const float *A, size_t lda,
              const void * /*QuantBData*/,
              const void *PackedQuantBDataWorkspace, const float *QuantBScale,
              const void *QuantBZeroPoint, const float *Bias, float *C,
              size_t ldc, void *Workspace,
              MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
              MLAS_THREADPOOL *Threadpool) {
  MLAS_SQNBIT_GEMM_DATA_PARAMS params;
  params.A = A;
  params.lda = lda;
  params.Bias = Bias;
  params.C = C;
  params.ldc = ldc;
#ifdef MLAS_TARGET_AMD64_IX86
  if (ComputeType == CompInt8) {
    params.QuantBDataWorkspace = PackedQuantBDataWorkspace;
  }
#endif
  params.PackedQuantBData =
    static_cast<const std::byte *>(PackedQuantBDataWorkspace);
  params.QuantBScale = QuantBScale;
  params.QuantBZeroPoint = QuantBZeroPoint;
  params.PostProcessor = nullptr;

  MlasSQNBitGemmBatch(M, N, K, 1, BlkBitWidth, BlkLen, ComputeType, &params,
                      Workspace, Threadpool);
}

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_get_gqu4_rhs_nt_t_quant_size(size_t N, size_t K,
                                           size_t &QuantBDataSizeInBytes,
                                           size_t &QuantBScaleSize,
                                           size_t &QuantBZeroPointSizeInBytes) {
  MlasBlockwiseQuantizedBufferSizes(BlkBitWidth, BlkLen, /* columnwise */ true,
                                    static_cast<int>(K), static_cast<int>(N),
                                    QuantBDataSizeInBytes, QuantBScaleSize,
                                    &QuantBZeroPointSizeInBytes);
}

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_gqu4_rhs_nt_t(const float *B, void *_QuantBData,
                            float *_QuantBScale, void *_QuantBZeroPoint,
                            size_t N, size_t K, bool Symmetric) {
  uint8_t *QuantBData = (uint8_t *)_QuantBData;
  float *QuantBScale = (float *)_QuantBScale;
  uint8_t *QuantBZeroPoint = (uint8_t *)_QuantBZeroPoint;
  MlasQuantizeBlockwise<float, BlkBitWidth>(
    QuantBData, QuantBScale, QuantBZeroPoint, B, BlkLen,
    /* columnwise */ true, static_cast<int>(K), static_cast<int>(N),
    static_cast<int>(N), GetMlasThreadPool());
}

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_gqu4_gemm(size_t M, size_t N, size_t K, const float *A,
                        size_t lda, const void *QuantBData,
                        const float *QuantBScale, const void *QuantBZeroPoint,
                        const float *Bias, float *C, size_t ldc,
                        MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
                        MLAS_THREADPOOL *Threadpool) {
  MatrixGuardBuffer<std::byte> BufferWorkspace;
  MatrixGuardBuffer<std::byte> BufferPackedQuantBData;
  auto t1 = high_resolution_clock::now();

  void *Workspace = nullptr;
  if (const auto WorkspaceSize = MlasSQNBitGemmBatchWorkspaceSize(
        M, N, K, 1, BlkBitWidth, BlkLen, ComputeType);
      WorkspaceSize > 0) {
    Workspace = BufferWorkspace.GetBuffer(WorkspaceSize);
  }

  void *PackedQuantBDataWorkspace = nullptr;
  if (const auto PackedQuantBDataSize = MlasSQNBitGemmPackQuantBDataSize(
        N, K, BlkBitWidth, BlkLen, ComputeType);
      PackedQuantBDataSize > 0) {
    PackedQuantBDataWorkspace =
      BufferPackedQuantBData.GetBuffer(PackedQuantBDataSize);

    bool has_zp_input = QuantBZeroPoint != nullptr;
    MlasSQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, ComputeType,
                                 QuantBData, PackedQuantBDataWorkspace,
                                 QuantBScale, has_zp_input, QuantBZeroPoint,
                                 GetMlasThreadPool());
  }

  CallGemm<BlkBitWidth, BlkLen>(
    M, N, K, A, /* lda */ K, QuantBData, PackedQuantBDataWorkspace, QuantBScale,
    QuantBZeroPoint, Bias, C, /* ldc */ N, Workspace, ComputeType, Threadpool);
}

template void nntr_sqn_gqu4_rhs_nt_t<4, 64>(const float *, void *, float *,
                                            void *, size_t, size_t, bool);

template void nntr_sqn_gqu4_gemm<4, 64>(size_t, size_t, size_t, const float *,
                                        size_t, const void *, const float *,
                                        const void *, const float *, float *,
                                        size_t, MLAS_SQNBIT_GEMM_COMPUTE_TYPE,
                                        MLAS_THREADPOOL *);

template void nntr_sqn_get_gqu4_rhs_nt_t_quant_size<4, 64>(size_t, size_t,
                                                           size_t &, size_t &,
                                                           size_t &);
