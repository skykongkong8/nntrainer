#include "mlas_q4.h"
#include "mlas_qnbit.h"

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_gqu4_rhs_nt_t(const float *B, const void *_QuantBData,
                               const float *_QuantBScale,
                               const void *_QuantBZeroPoint, size_t N, size_t K,
                               bool Symmetric);

template <size_t BlkBitWidth, size_t BlkLen>
void nntr_sqn_gqu4_gemm(size_t M, size_t N, size_t K, const float *A, size_t lda,
                   const void *QuantBData,
                   const float *QuantBScale, const void *QuantBZeroPoint,
                   const float *Bias, float *C, size_t ldc,
                   MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType = CompFp32,
                   MLAS_THREADPOOL *Threadpool = nullptr);


