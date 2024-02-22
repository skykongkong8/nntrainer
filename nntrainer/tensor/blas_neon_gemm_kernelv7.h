#include <blas_neon_gemm_setting.h>

static inline void hgemmv7_opt(unsigned int M, unsigned int N, unsigned int K,
                               float alpha, const __fp16 *A, unsigned int LDA,
                               const __fp16 *B, unsigned int LDB, float beta,
                               __fp16 *C, unsigned int LDC) {
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


void hgemmv7(unsigned int M, unsigned int N, unsigned int K, float alpha,
             __fp16 *A, unsigned int LDA, __fp16 *B, unsigned int LDB,
             float beta, __fp16 *C, unsigned int LDC) {
  unsigned int i, j, k;
  /// @todo do for beta
  // if (beta != 1.0) return;
  unsigned int M_ = M & -VL_FP16, N_ = N & -VL_FP16_HALF,
               K_ = K & -VL_FP16_HALF;
  /// @todo do for alpha
  float16x8_t valpha = vdupq_n_f16(1.f); // broadcast alpha to a 256-bit vector
  float16x8_t a, b0, b1, b2, b3;
  for (i = 0; i < M_; i += VL_FP16) {
    for (j = 0; j < N_; j += VL_FP16_HALF) {
      float16x8_t c0 = vmovq_n_f16(0.f);
      float16x8_t c1 = vmovq_n_f16(0.f);
      float16x8_t c2 = vmovq_n_f16(0.f);
      float16x8_t c3 = vmovq_n_f16(0.f);
      for (k = 0; k < K_;) {
        KERNEL_8x4_HGEMM();
        KERNEL_8x4_HGEMM();
        KERNEL_8x4_HGEMM();
        KERNEL_8x4_HGEMM();
      }
      for (k = K_; k < K;) {
        KERNEL_8x4_HGEMM();
      }
      vst1q_f16(&C(i, j), vaddq_f16(c0, vld1q_f16(&C(i, j))));
      vst1q_f16(&C(i, j + 1), vaddq_f16(c1, vld1q_f16(&C(i, j + 1))));
      vst1q_f16(&C(i, j + 2), vaddq_f16(c2, vld1q_f16(&C(i, j + 2))));
      vst1q_f16(&C(i, j + 3), vaddq_f16(c3, vld1q_f16(&C(i, j + 3))));
    }
  }
  if (M_ == M && N_ == N)
    return;
  // boundary conditions
  if (M_ != M)
    hgemmv7_opt(M - M_, N, K, alpha, A + M_, LDA, B, LDB, 1.0, &C(M_, 0), LDC);
  if (N_ != N)
    hgemmv7_opt(M_, N - N_, K, alpha, A, LDA, &B(0, N_), LDB, 1.0, &C(0, N_),
                LDC);
}