#include <sqnbitgemm_interface.h>
#include <test_util.h>
#include <iostream>

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
void nntr_sqn_gqu4_rhs_nt_t(const float *B, void *_QuantBData,
                            float *_QuantBScale,
                            void *_QuantBZeroPoint, size_t N, size_t K,
                            bool Symmetric) {
  MatrixGuardBuffer<uint8_t> BufferQuantBData;
  MatrixGuardBuffer<float> BufferQuantBScale;
  MatrixGuardBuffer<uint8_t> BufferQuantBZeroPoint;

  uint8_t *QuantBData = (uint8_t *)_QuantBData;
  float *QuantBScale = (float *)_QuantBScale;
  uint8_t *QuantBZeroPoint = (uint8_t *)_QuantBZeroPoint;
  {
    size_t QuantBDataSizeInBytes, QuantBScaleSize, QuantBZeroPointSizeInBytes;
    MlasBlockwiseQuantizedBufferSizes(
      BlkBitWidth, BlkLen, /* columnwise */ true, static_cast<int>(K),
      static_cast<int>(N), QuantBDataSizeInBytes, QuantBScaleSize,
      &QuantBZeroPointSizeInBytes);

    QuantBData = BufferQuantBData.GetBuffer(QuantBDataSizeInBytes);
    QuantBScale = BufferQuantBScale.GetBuffer(QuantBScaleSize);
    if (!Symmetric) {
      QuantBZeroPoint =
        BufferQuantBZeroPoint.GetBuffer(QuantBZeroPointSizeInBytes);
    }

    MlasQuantizeBlockwise<float, BlkBitWidth>(
      QuantBData, QuantBScale, QuantBZeroPoint, B, BlkLen,
      /* columnwise */ true, static_cast<int>(K), static_cast<int>(N),
      static_cast<int>(N), GetMlasThreadPool());
  }
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

  void *Workspace = nullptr;
  if (const auto WorkspaceSize = MlasSQNBitGemmBatchWorkspaceSize(
        M, N, K, 1, BlkBitWidth, BlkLen, ComputeType);
      WorkspaceSize > 0) {
    Workspace = BufferWorkspace.GetBuffer(WorkspaceSize);
  }
  std::cout << "MlasSQNBitGemmBatchWorkspaceSize\n";


  void *PackedQuantBDataWorkspace = nullptr;
  if (const auto PackedQuantBDataSize = MlasSQNBitGemmPackQuantBDataSize(
        N, K, BlkBitWidth, BlkLen, ComputeType);
      PackedQuantBDataSize > 0) {
    PackedQuantBDataWorkspace =
      BufferPackedQuantBData.GetBuffer(PackedQuantBDataSize);

  std::cout << "MlasSQNBitGemmPackQuantBDataSize\n";

    bool has_zp_input = QuantBZeroPoint != nullptr;
    MlasSQNBitGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, ComputeType,
                                 QuantBData, PackedQuantBDataWorkspace,
                                 QuantBScale, has_zp_input, QuantBZeroPoint,
                                 GetMlasThreadPool());
  }

  std::cout << "MlasSQNBitGemmPackQuantBData\n";

  CallGemm<BlkBitWidth, BlkLen>(
    M, N, K, A, /* lda */ K, QuantBData, PackedQuantBDataWorkspace, QuantBScale,
    QuantBZeroPoint, Bias, C, /* ldc */ N, Workspace, ComputeType, Threadpool);
  std::cout << "CallGemm\n";

}

template void nntr_sqn_gqu4_rhs_nt_t<4, 64>(const float *, void *,
                                                   float *, void *,
                                                   size_t, size_t, bool);

template void nntr_sqn_gqu4_gemm<4, 64>(size_t, size_t, size_t, const float *,
                                        size_t, const void *, const float *,
                                        const void *, const float *, float *,
                                        size_t, MLAS_SQNBIT_GEMM_COMPUTE_TYPE,
                                        MLAS_THREADPOOL *);
