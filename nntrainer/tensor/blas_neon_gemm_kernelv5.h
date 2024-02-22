#include <blas_neon_gemm_setting.h>

static inline void hgemmv5_opt(unsigned int M, unsigned int N,
                                      unsigned int K, float alpha,
                                      const __fp16 *A, unsigned int LDA,
                                      const __fp16 *B, unsigned int LDB,
                                      float beta, __fp16 *C, unsigned int LDC) {
  unsigned int i, j, k;
  /// @todo do for beta
  // if (beta != 1.0) return;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      __fp16 tmp = C(i, j);
      // float tmp=C(i,j);
      for (k = 0; k < K; k++) {
        tmp += alpha * A(i, k) * B(k, j);
      }
      C(i, j) = tmp;
    }
  }
}

void hgemmv5(unsigned int M, unsigned int N, unsigned int K, float alpha,
                    __fp16 *A, unsigned int LDA, __fp16 *B, unsigned int LDB,
                    float beta, __fp16 *C, unsigned int LDC) {
  unsigned int i, j, k;
  /// @todo do for beta
  // if (beta != 1.0) return;
  unsigned int M_ = M & -VL_FP16_HALF, N_ = N & -VL_FP16_HALF;
  /// @todo do for alpha
  float16x4_t valpha = vdup_n_f16(1.f); // broadcast alpha to a 256-bit vector
  for (i = 0; i < M_; i += VL_FP16_HALF) {
    for (j = 0; j < N_; j += VL_FP16_HALF) {
      float16x4_t c0 = vmov_n_f16(0.f);
      float16x4_t c1 = vmov_n_f16(0.f);
      float16x4_t c2 = vmov_n_f16(0.f);
      float16x4_t c3 = vmov_n_f16(0.f);
      for (k = 0; k < K; k++) {
        float16x4_t a = vmul_f16(valpha, vld1_f16(&A(i, k)));
        float16x4_t b0 = vmov_n_f16(B(k, j));
        float16x4_t b1 = vmov_n_f16(B(k, j + 1));
        float16x4_t b2 = vmov_n_f16(B(k, j + 2));
        float16x4_t b3 = vmov_n_f16(B(k, j + 3));
        c0 = vfma_f16(c0, b0, a);
        c1 = vfma_f16(c1, b1, a);
        c2 = vfma_f16(c2, b2, a);
        c3 = vfma_f16(c3, b3, a);
      }
      vst1_f16(&C(i, j), vadd_f16(c0, vld1_f16(&C(i, j))));
      vst1_f16(&C(i, j + 1), vadd_f16(c1, vld1_f16(&C(i, j + 1))));
      vst1_f16(&C(i, j + 2), vadd_f16(c2, vld1_f16(&C(i, j + 2))));
      vst1_f16(&C(i, j + 3), vadd_f16(c3, vld1_f16(&C(i, j + 3))));
    }
  }
  if (M_ == M && N_ == N)
    return;
  // boundary conditions
  if (M_ != M)
    hgemmv5_opt(M - M_, N, K, alpha, A + M_, LDA, B, LDB, 1.0, &C(M_, 0),
                       LDC);
  if (N_ != N)
    hgemmv5_opt(M_, N - N_, K, alpha, A, LDA, &B(0, N_), LDB, 1.0,
                       &C(0, N_), LDC);
}