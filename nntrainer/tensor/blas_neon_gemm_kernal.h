#include <arm_neon.h>
#include <omp.h>
#include <iostream>
#include <blas_neon_gemm_kernelv5.h>
#include <blas_neon_gemm_kernelv6.h>
#include <blas_neon_gemm_kernelv7.h>
#include <blas_neon_gemm_kernelv8.h>
#include <blas_neon_gemm_kernelv9.h>
#include <blas_neon_gemm_kernelv9_2.h>
#include <blas_neon_gemm_kernelv10.h>
#include <blas_neon_gemm_kernelv11.h>


inline void matrix_kernel8_6(const float *A, const float *B, float *C,
                             const int M, const int N, const int K, const int m,
                             const int n) {
  float32x4_t mB0, mB1; // 4 lanes of 32-bit (FP32) data
  float32x4_t mA0, mA1;

  // Chose kernel size 6x8
  // - 8 because SIMD width is 4*32 (so must be multiple of 4)
  // - Also overall 16 registers
  // - Number of registers depends on NEON (ARMv7, ARMv8, etc.)
  // - So having 6x8 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  float32x4_t result0_0 = vdupq_n_f32(0.0f); // Initialize all 4 lanes to 0
  float32x4_t result1_0 = vdupq_n_f32(0.0f);
  float32x4_t result2_0 = vdupq_n_f32(0.0f);
  float32x4_t result3_0 = vdupq_n_f32(0.0f);
  float32x4_t result4_0 = vdupq_n_f32(0.0f);
  float32x4_t result5_0 = vdupq_n_f32(0.0f);
  float32x4_t result0_1 = vdupq_n_f32(0.0f);
  float32x4_t result1_1 = vdupq_n_f32(0.0f);
  float32x4_t result2_1 = vdupq_n_f32(0.0f);
  float32x4_t result3_1 = vdupq_n_f32(0.0f);
  float32x4_t result4_1 = vdupq_n_f32(0.0f);
  float32x4_t result5_1 = vdupq_n_f32(0.0f);

  // This is the same for loop as in naive implementation, except now instead of
  // the k indexing a single dot product of 2 vectors of size k (a row of A and
  // a col of B), the k is indexing 6 rows of A and 16 cols of B Since the SIMD
  // width is 4 (128 bits), need to do 8 fmas here
  for (int k = 0; k < K; k++) {
    // Load the k'th row of the B block (load twice since in total, it's 8
    // floats)
    mB0 = vld1q_f32(&B[N * k + n + 4 * 0]);
    mB1 = vld1q_f32(&B[N * k + n + 4 * 1]);

    // Load a single value for the k'th col of A
    // In total, we need to do this 6 times (col of A has height 6)
    mA0 = vdupq_n_f32(
      A[k + (m + 0) * K]); // Load float @ A's col k, row m+0 into reg
    mA1 = vdupq_n_f32(A[k + (m + 1) * K]); // Load float @ A's col k, row m+1

    // Perform FMAs
    result0_0 = vfmaq_f32(result0_0, mB0, mA0);
    result0_1 = vfmaq_f32(result0_1, mB1, mA0);
    result1_0 = vfmaq_f32(result1_0, mB0, mA1);
    result1_1 = vfmaq_f32(result1_1, mB1, mA1);

    // Repeat for the other 4

    mA0 = vdupq_n_f32(A[k + (m + 2) * K]);
    mA1 = vdupq_n_f32(A[k + (m + 3) * K]);
    result2_0 = vfmaq_f32(result2_0, mB0, mA0);
    result2_1 = vfmaq_f32(result2_1, mB1, mA0);
    result3_0 = vfmaq_f32(result3_0, mB0, mA1);
    result3_1 = vfmaq_f32(result3_1, mB1, mA1);

    mA0 = vdupq_n_f32(A[k + (m + 4) * K]);
    mA1 = vdupq_n_f32(A[k + (m + 5) * K]);
    result4_0 = vfmaq_f32(result4_0, mB0, mA0);
    result4_1 = vfmaq_f32(result4_1, mB1, mA0);
    result5_0 = vfmaq_f32(result5_0, mB0, mA1);
    result5_1 = vfmaq_f32(result5_1, mB1, mA1);
  }

  // Write registers back to C
  vst1q_f32(&C[(m + 0) * N + n + 0 * 4], result0_0);
  vst1q_f32(&C[(m + 1) * N + n + 0 * 4], result1_0);
  vst1q_f32(&C[(m + 2) * N + n + 0 * 4], result2_0);
  vst1q_f32(&C[(m + 3) * N + n + 0 * 4], result3_0);
  vst1q_f32(&C[(m + 4) * N + n + 0 * 4], result4_0);
  vst1q_f32(&C[(m + 5) * N + n + 0 * 4], result5_0);
  vst1q_f32(&C[(m + 0) * N + n + 1 * 4], result0_1);
  vst1q_f32(&C[(m + 1) * N + n + 1 * 4], result1_1);
  vst1q_f32(&C[(m + 2) * N + n + 1 * 4], result2_1);
  vst1q_f32(&C[(m + 3) * N + n + 1 * 4], result3_1);
  vst1q_f32(&C[(m + 4) * N + n + 1 * 4], result4_1);
  vst1q_f32(&C[(m + 5) * N + n + 1 * 4], result5_1);
}

#ifdef ENABLE_FP16
inline void matrix_kernel16_6(const __fp16 *A, const __fp16 *B, float *C32,
                              uint32_t M, uint32_t N, uint32_t K, uint32_t m,
                              uint32_t n) {

  float16x8_t mB0, mB1; // 4 lanes of 32-bit (FP32) data
  float16x8_t mA0, mA1;

  // Chose kernel size 6x8
  // - 8 because SIMD width is 4*32 (so must be multiple of 4)
  // - Also overall 16 registers
  // - Number of registers depends on NEON (ARMv7, ARMv8, etc.)
  // - So having 6x8 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  float16x8_t result0_0 = vdupq_n_f16(0.0f); // Initialize all 4 lanes to 0
  float16x8_t result1_0 = vdupq_n_f16(0.0f);
  float16x8_t result2_0 = vdupq_n_f16(0.0f);
  float16x8_t result3_0 = vdupq_n_f16(0.0f);
  float16x8_t result4_0 = vdupq_n_f16(0.0f);
  float16x8_t result5_0 = vdupq_n_f16(0.0f);
  float16x8_t result0_1 = vdupq_n_f16(0.0f);
  float16x8_t result1_1 = vdupq_n_f16(0.0f);
  float16x8_t result2_1 = vdupq_n_f16(0.0f);
  float16x8_t result3_1 = vdupq_n_f16(0.0f);
  float16x8_t result4_1 = vdupq_n_f16(0.0f);
  float16x8_t result5_1 = vdupq_n_f16(0.0f);

  // This is the same for loop as in naive implementation, except now instead of
  // the k indexing a single dot product of 2 vectors of size k (a row of A and
  // a col of B), the k is indexing 6 rows of A and 16 cols of B Since the SIMD
  // width is 4 (128 bits), need to do 8 fmas here

#pragma omp parallel for schedule(guided) num_threads(2)
  for (unsigned int k = 0; k < K; k++) {
    // Load the k'th row of the B block (load twice since in total, it's 8
    // floats)
    mB0 = vld1q_f16(&B[N * k + n + 8 * 0]);
    mB1 = vld1q_f16(&B[N * k + n + 8 * 1]);

    // Load a single value for the k'th col of A
    // In total, we need to do this 6 times (col of A has height 6)
    mA0 = vdupq_n_f16(
      A[k + (m + 0) * K]); // Load float @ A's col k, row m+0 into reg
    mA1 = vdupq_n_f16(A[k + (m + 1) * K]); // Load float @ A's col k, row m+1

    // Perform FMAs
    result0_0 = vfmaq_f16(result0_0, mB0, mA0);
    result0_1 = vfmaq_f16(result0_1, mB1, mA0);
    result1_0 = vfmaq_f16(result1_0, mB0, mA1);
    result1_1 = vfmaq_f16(result1_1, mB1, mA1);

    // Repeat for the other 4

    mA0 = vdupq_n_f16(A[k + (m + 2) * K]);
    mA1 = vdupq_n_f16(A[k + (m + 3) * K]);
    result2_0 = vfmaq_f16(result2_0, mB0, mA0);
    result2_1 = vfmaq_f16(result2_1, mB1, mA0);
    result3_0 = vfmaq_f16(result3_0, mB0, mA1);
    result3_1 = vfmaq_f16(result3_1, mB1, mA1);

    mA0 = vdupq_n_f16(A[k + (m + 4) * K]);
    mA1 = vdupq_n_f16(A[k + (m + 5) * K]);
    result4_0 = vfmaq_f16(result4_0, mB0, mA0);
    result4_1 = vfmaq_f16(result4_1, mB1, mA0);
    result5_0 = vfmaq_f16(result5_0, mB0, mA1);
    result5_1 = vfmaq_f16(result5_1, mB1, mA1);
  }

  // Write registers back to C
  vst1q_f32(&C32[(m + 0) * N + n + 0 * 8],
            vcvt_f32_f16(vget_low_f16(result0_0)));
  vst1q_f32(&C32[(m + 1) * N + n + 0 * 8],
            vcvt_f32_f16(vget_low_f16(result1_0)));
  vst1q_f32(&C32[(m + 2) * N + n + 0 * 8],
            vcvt_f32_f16(vget_low_f16(result2_0)));
  vst1q_f32(&C32[(m + 3) * N + n + 0 * 8],
            vcvt_f32_f16(vget_low_f16(result3_0)));
  vst1q_f32(&C32[(m + 4) * N + n + 0 * 8],
            vcvt_f32_f16(vget_low_f16(result4_0)));
  vst1q_f32(&C32[(m + 5) * N + n + 0 * 8],
            vcvt_f32_f16(vget_low_f16(result5_0)));

  vst1q_f32(&C32[(m + 0) * N + n + 0 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result0_0)));
  vst1q_f32(&C32[(m + 1) * N + n + 0 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result1_0)));
  vst1q_f32(&C32[(m + 2) * N + n + 0 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result2_0)));
  vst1q_f32(&C32[(m + 3) * N + n + 0 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result3_0)));
  vst1q_f32(&C32[(m + 4) * N + n + 0 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result4_0)));
  vst1q_f32(&C32[(m + 5) * N + n + 0 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result5_0)));

  vst1q_f32(&C32[(m + 0) * N + n + 1 * 8],
            vcvt_f32_f16(vget_low_f16(result0_1)));
  vst1q_f32(&C32[(m + 1) * N + n + 1 * 8],
            vcvt_f32_f16(vget_low_f16(result1_1)));
  vst1q_f32(&C32[(m + 2) * N + n + 1 * 8],
            vcvt_f32_f16(vget_low_f16(result2_1)));
  vst1q_f32(&C32[(m + 3) * N + n + 1 * 8],
            vcvt_f32_f16(vget_low_f16(result3_1)));
  vst1q_f32(&C32[(m + 4) * N + n + 1 * 8],
            vcvt_f32_f16(vget_low_f16(result4_1)));
  vst1q_f32(&C32[(m + 5) * N + n + 1 * 8],
            vcvt_f32_f16(vget_low_f16(result5_1)));

  vst1q_f32(&C32[(m + 0) * N + n + 1 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result0_1)));
  vst1q_f32(&C32[(m + 1) * N + n + 1 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result1_1)));
  vst1q_f32(&C32[(m + 2) * N + n + 1 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result2_1)));
  vst1q_f32(&C32[(m + 3) * N + n + 1 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result3_1)));
  vst1q_f32(&C32[(m + 4) * N + n + 1 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result4_1)));
  vst1q_f32(&C32[(m + 5) * N + n + 1 * 8 + 4],
            vcvt_f32_f16(vget_high_f16(result5_1)));
}

inline void matrix_kernel16_6(const __fp16 *A, const __fp16 *B, __fp16 *C,
                              uint32_t M, uint32_t N, uint32_t K, uint32_t m,
                              uint32_t n) {

  float16x8_t mB0, mB1; // 4 lanes of 32-bit (FP32) data
  float16x8_t mA0, mA1;

  // Chose kernel size 6x8
  // - 8 because SIMD width is 4*32 (so must be multiple of 4)
  // - Also overall 16 registers
  // - Number of registers depends on NEON (ARMv7, ARMv8, etc.)
  // - So having 6x8 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  float16x8_t result0_0 = vdupq_n_f16(0.0f); // Initialize all 4 lanes to 0
  float16x8_t result1_0 = vdupq_n_f16(0.0f);
  float16x8_t result2_0 = vdupq_n_f16(0.0f);
  float16x8_t result3_0 = vdupq_n_f16(0.0f);
  float16x8_t result4_0 = vdupq_n_f16(0.0f);
  float16x8_t result5_0 = vdupq_n_f16(0.0f);
  float16x8_t result0_1 = vdupq_n_f16(0.0f);
  float16x8_t result1_1 = vdupq_n_f16(0.0f);
  float16x8_t result2_1 = vdupq_n_f16(0.0f);
  float16x8_t result3_1 = vdupq_n_f16(0.0f);
  float16x8_t result4_1 = vdupq_n_f16(0.0f);
  float16x8_t result5_1 = vdupq_n_f16(0.0f);

  // This is the same for loop as in naive implementation, except now instead of
  // the k indexing a single dot product of 2 vectors of size k (a row of A and
  // a col of B), the k is indexing 6 rows of A and 16 cols of B Since the SIMD
  // width is 4 (128 bits), need to do 8 fmas here

  // #pragma omp parallel for schedule(guided) num_threads(NEON_NUM_THREADS)
  for (unsigned int k = 0; k < K; k++) {
    // Load the k'th row of the B block (load twice since in total, it's 8
    // floats)
    mB0 = vld1q_f16(&B[N * k + n + 8 * 0]);
    mB1 = vld1q_f16(&B[N * k + n + 8 * 1]);

    // Load a single value for the k'th col of A
    // In total, we need to do this 6 times (col of A has height 6)
    mA0 = vdupq_n_f16(
      A[k + (m + 0) * K]); // Load float @ A's col k, row m+0 into reg
    mA1 = vdupq_n_f16(A[k + (m + 1) * K]); // Load float @ A's col k, row m+1

    // Perform FMAs
    result0_0 = vfmaq_f16(result0_0, mB0, mA0);
    result0_1 = vfmaq_f16(result0_1, mB1, mA0);
    result1_0 = vfmaq_f16(result1_0, mB0, mA1);
    result1_1 = vfmaq_f16(result1_1, mB1, mA1);

    // Repeat for the other 4

    mA0 = vdupq_n_f16(A[k + (m + 2) * K]);
    mA1 = vdupq_n_f16(A[k + (m + 3) * K]);
    result2_0 = vfmaq_f16(result2_0, mB0, mA0);
    result2_1 = vfmaq_f16(result2_1, mB1, mA0);
    result3_0 = vfmaq_f16(result3_0, mB0, mA1);
    result3_1 = vfmaq_f16(result3_1, mB1, mA1);

    mA0 = vdupq_n_f16(A[k + (m + 4) * K]);
    mA1 = vdupq_n_f16(A[k + (m + 5) * K]);
    result4_0 = vfmaq_f16(result4_0, mB0, mA0);
    result4_1 = vfmaq_f16(result4_1, mB1, mA0);
    result5_0 = vfmaq_f16(result5_0, mB0, mA1);
    result5_1 = vfmaq_f16(result5_1, mB1, mA1);
  }

  // Write registers back to C
  vst1q_f16(&C[(m + 0) * N + n + 0 * 8], result0_0);
  vst1q_f16(&C[(m + 1) * N + n + 0 * 8], result1_0);
  vst1q_f16(&C[(m + 2) * N + n + 0 * 8], result2_0);
  vst1q_f16(&C[(m + 3) * N + n + 0 * 8], result3_0);
  vst1q_f16(&C[(m + 4) * N + n + 0 * 8], result4_0);
  vst1q_f16(&C[(m + 5) * N + n + 0 * 8], result5_0);

  vst1q_f16(&C[(m + 0) * N + n + 1 * 8], result0_1);
  vst1q_f16(&C[(m + 1) * N + n + 1 * 8], result1_1);
  vst1q_f16(&C[(m + 2) * N + n + 1 * 8], result2_1);
  vst1q_f16(&C[(m + 3) * N + n + 1 * 8], result3_1);
  vst1q_f16(&C[(m + 4) * N + n + 1 * 8], result4_1);
  vst1q_f16(&C[(m + 5) * N + n + 1 * 8], result5_1);
}

inline void matrix_kernel16_6(const uint16_t *A, const uint16_t *B, uint16_t *C,
                              uint32_t M, uint32_t N, uint32_t K, uint32_t m,
                              uint32_t n) {

  uint16x8_t mB0, mB1; // 4 lanes of 32-bit (FP32) data
  uint16x8_t mA0, mA1;

  // Chose kernel size 6x8
  // - 8 because SIMD width is 4*32 (so must be multiple of 4)
  // - Also overall 16 registers
  // - Number of registers depends on NEON (ARMv7, ARMv8, etc.)
  // - So having 6x8 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  uint16x8_t result0_0 = vdupq_n_u16(0.0f); // Initialize all 4 lanes to 0
  uint16x8_t result1_0 = vdupq_n_u16(0.0f);
  uint16x8_t result2_0 = vdupq_n_u16(0.0f);
  uint16x8_t result3_0 = vdupq_n_u16(0.0f);
  uint16x8_t result4_0 = vdupq_n_u16(0.0f);
  uint16x8_t result5_0 = vdupq_n_u16(0.0f);
  uint16x8_t result0_1 = vdupq_n_u16(0.0f);
  uint16x8_t result1_1 = vdupq_n_u16(0.0f);
  uint16x8_t result2_1 = vdupq_n_u16(0.0f);
  uint16x8_t result3_1 = vdupq_n_u16(0.0f);
  uint16x8_t result4_1 = vdupq_n_u16(0.0f);
  uint16x8_t result5_1 = vdupq_n_u16(0.0f);

  // This is the same for loop as in naive implementation, except now instead of
  // the k indexing a single dot product of 2 vectors of size k (a row of A and
  // a col of B), the k is indexing 6 rows of A and 16 cols of B Since the SIMD
  // width is 4 (128 bits), need to do 8 fmas here

  // #pragma omp parallel for schedule(guided) num_threads(2)
  for (unsigned int k = 0; k < K; k++) {
    // Load the k'th row of the B block (load twice since in total, it's 8
    // floats)
    mB0 = vld1q_u16(&B[N * k + n + 8 * 0]);
    mB1 = vld1q_u16(&B[N * k + n + 8 * 1]);

    // Load a single value for the k'th col of A
    // In total, we need to do this 6 times (col of A has height 6)
    mA0 = vdupq_n_u16(
      A[k + (m + 0) * K]); // Load float @ A's col k, row m+0 into reg
    mA1 = vdupq_n_u16(A[k + (m + 1) * K]); // Load float @ A's col k, row m+1

    // Perform FMAs
    result0_0 = vmlaq_u16(result0_0, mB0, mA0);
    result0_1 = vmlaq_u16(result0_1, mB1, mA0);
    result1_0 = vmlaq_u16(result1_0, mB0, mA1);
    result1_1 = vmlaq_u16(result1_1, mB1, mA1);

    // Repeat for the other 4

    mA0 = vdupq_n_u16(A[k + (m + 2) * K]);
    mA1 = vdupq_n_u16(A[k + (m + 3) * K]);
    result2_0 = vmlaq_u16(result2_0, mB0, mA0);
    result2_1 = vmlaq_u16(result2_1, mB1, mA0);
    result3_0 = vmlaq_u16(result3_0, mB0, mA1);
    result3_1 = vmlaq_u16(result3_1, mB1, mA1);

    mA0 = vdupq_n_u16(A[k + (m + 4) * K]);
    mA1 = vdupq_n_u16(A[k + (m + 5) * K]);
    result4_0 = vmlaq_u16(result4_0, mB0, mA0);
    result4_1 = vmlaq_u16(result4_1, mB1, mA0);
    result5_0 = vmlaq_u16(result5_0, mB0, mA1);
    result5_1 = vmlaq_u16(result5_1, mB1, mA1);
  }

  // Write registers back to C
  vst1q_u16(&C[(m + 0) * N + n + 0 * 8], result0_0);
  vst1q_u16(&C[(m + 1) * N + n + 0 * 8], result1_0);
  vst1q_u16(&C[(m + 2) * N + n + 0 * 8], result2_0);
  vst1q_u16(&C[(m + 3) * N + n + 0 * 8], result3_0);
  vst1q_u16(&C[(m + 4) * N + n + 0 * 8], result4_0);
  vst1q_u16(&C[(m + 5) * N + n + 0 * 8], result5_0);

  vst1q_u16(&C[(m + 0) * N + n + 1 * 8], result0_1);
  vst1q_u16(&C[(m + 1) * N + n + 1 * 8], result1_1);
  vst1q_u16(&C[(m + 2) * N + n + 1 * 8], result2_1);
  vst1q_u16(&C[(m + 3) * N + n + 1 * 8], result3_1);
  vst1q_u16(&C[(m + 4) * N + n + 1 * 8], result4_1);
  vst1q_u16(&C[(m + 5) * N + n + 1 * 8], result5_1);
}

inline void matrix_kernel16_6_block_res16(
  const __fp16 *A, const __fp16 *B, float *C32, const int M, const int N,
  const int K, const int jc, const int nc, const int pc, const int kc,
  const int ic, const int mc, const int jr, const int nr, const int ir,
  const int mr) {
  float16x8_t mB0, mB1;
  float16x8_t mA0, mA1;

  // Chose kernel size 6x16
  // - 16 because SIMD width is 4*32 (so must be multiple of 4)
  // - Also overall 16 registers
  // - Number of registers depends on NEON (ARMv7, ARMv8, etc.)
  // - So having 6x16 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  float16x8_t result0_0 = vdupq_n_f16(0.0f); // Initialize all 4 lanes to 0
  float16x8_t result1_0 = vdupq_n_f16(0.0f);
  float16x8_t result2_0 = vdupq_n_f16(0.0f);
  float16x8_t result3_0 = vdupq_n_f16(0.0f);
  float16x8_t result4_0 = vdupq_n_f16(0.0f);
  float16x8_t result5_0 = vdupq_n_f16(0.0f);
  float16x8_t result0_1 = vdupq_n_f16(0.0f);
  float16x8_t result1_1 = vdupq_n_f16(0.0f);
  float16x8_t result2_1 = vdupq_n_f16(0.0f);
  float16x8_t result3_1 = vdupq_n_f16(0.0f);
  float16x8_t result4_1 = vdupq_n_f16(0.0f);
  float16x8_t result5_1 = vdupq_n_f16(0.0f);

  // This is the same for loop as in naive implementation, except now instead of
  // the k indexing a single dot product of 2 vectors of size k (a row of A and
  // a col of B), the k is indexing 6 rows of A and 16 cols of B Since the SIMD
  // width is 4 (128 bits), need to do 8 fmas here
  for (int k = 0; k < kc; ++k) {
    // Load the k'th row of the B block (load twice since in total, it's 16
    // floats)
    mB0 = vld1q_f16(&B[N * (k + pc) + jc + jr + 8 * 0]);
    mB1 = vld1q_f16(&B[N * (k + pc) + jc + jr + 8 * 1]);

    // Load a single value for the k'th col of A
    // In total, we need to do this 6 times (col of A has height 6)
    mA0 = vdupq_n_f16(A[k + pc + (ic + ir + 0) * K]); // Load float @ A's col k,
                                                      // row m+0 into reg
    mA1 = vdupq_n_f16(
      A[k + pc + (ic + ir + 1) * K]); // Load float @ A's col k, row m+1

    // Perform FMAs
    result0_0 = vfmaq_f16(result0_0, mB0, mA0);
    result0_1 = vfmaq_f16(result0_1, mB1, mA0);
    result1_0 = vfmaq_f16(result1_0, mB0, mA1);
    result1_1 = vfmaq_f16(result1_1, mB1, mA1);

    // Repeat for the other 4

    mA0 = vdupq_n_f16(A[k + pc + (ic + ir + 2) * K]);
    mA1 = vdupq_n_f16(A[k + pc + (ic + ir + 3) * K]);
    result2_0 = vfmaq_f16(result2_0, mB0, mA0);
    result2_1 = vfmaq_f16(result2_1, mB1, mA0);
    result3_0 = vfmaq_f16(result3_0, mB0, mA1);
    result3_1 = vfmaq_f16(result3_1, mB1, mA1);

    mA0 = vdupq_n_f16(A[k + pc + (ic + ir + 4) * K]);
    mA1 = vdupq_n_f16(A[k + pc + (ic + ir + 5) * K]);
    result4_0 = vfmaq_f16(result4_0, mB0, mA0);
    result4_1 = vfmaq_f16(result4_1, mB1, mA0);
    result5_0 = vfmaq_f16(result5_0, mB0, mA1);
    result5_1 = vfmaq_f16(result5_1, mB1, mA1);
  }

  // Write registers back to C
  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result0_0))));
  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result0_1))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result1_0))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result1_1))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result2_0))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result2_1))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result3_0))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result3_1))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result4_0))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result4_1))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result5_0))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result5_1))));

  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result0_0))));
  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result0_1))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result1_0))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result1_1))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result2_0))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result2_1))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result3_0))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result3_1))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result4_0))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result4_1))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result5_0))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result5_1))));
}

inline void matrix_kernel16_6_block(const __fp16 *A, const __fp16 *B,
                                    float *C32, const int M, const int N,
                                    const int K, const int jc, const int nc,
                                    const int pc, const int kc, const int ic,
                                    const int mc, const int jr, const int nr,
                                    const int ir, const int mr) {
  float16x8_t mB0, mB1;
  float16x8_t mA0, mA1;

  // Chose kernel size 6x16
  // - 16 because SIMD width is 4*32 (so must be multiple of 4)
  // - Also overall 16 registers
  // - Number of registers depends on NEON (ARMv7, ARMv8, etc.)
  // - So having 6x16 means 6x2 registers used for C block
  // - This leaves 4 for sections of A and B (needed to do fma)
  float16x8_t result0_0 = vdupq_n_f16(0.0f); // Initialize all 4 lanes to 0
  float16x8_t result1_0 = vdupq_n_f16(0.0f);
  float16x8_t result2_0 = vdupq_n_f16(0.0f);
  float16x8_t result3_0 = vdupq_n_f16(0.0f);
  float16x8_t result4_0 = vdupq_n_f16(0.0f);
  float16x8_t result5_0 = vdupq_n_f16(0.0f);
  float16x8_t result0_1 = vdupq_n_f16(0.0f);
  float16x8_t result1_1 = vdupq_n_f16(0.0f);
  float16x8_t result2_1 = vdupq_n_f16(0.0f);
  float16x8_t result3_1 = vdupq_n_f16(0.0f);
  float16x8_t result4_1 = vdupq_n_f16(0.0f);
  float16x8_t result5_1 = vdupq_n_f16(0.0f);

  // This is the same for loop as in naive implementation, except now instead of
  // the k indexing a single dot product of 2 vectors of size k (a row of A and
  // a col of B), the k is indexing 6 rows of A and 16 cols of B Since the SIMD
  // width is 4 (128 bits), need to do 8 fmas here

  // problem : accumulate for all K
  // current neon gemm : accumulate only for every 16~32 K, not all K
  for (int k = 0; k < kc; ++k) {
    // Load the k'th row of the B block (load twice since in total, it's 16
    // floats)
    mB0 = vld1q_f16(&B[N * (k + pc) + jc + jr + 8 * 0]);
    mB1 = vld1q_f16(&B[N * (k + pc) + jc + jr + 8 * 1]);

    // Load a single value for the k'th col of A
    // In total, we need to do this 6 times (col of A has height 6)
    mA0 = vdupq_n_f16(A[k + pc + (ic + ir + 0) * K]); // Load float @ A's col k,
                                                      // row m+0 into reg
    mA1 = vdupq_n_f16(
      A[k + pc + (ic + ir + 1) * K]); // Load float @ A's col k, row m+1

    // Perform FMAs
    result0_0 = vfmaq_f16(result0_0, mB0, mA0);
    result0_1 = vfmaq_f16(result0_1, mB1, mA0);
    result1_0 = vfmaq_f16(result1_0, mB0, mA1);
    result1_1 = vfmaq_f16(result1_1, mB1, mA1);

    // Repeat for the other 4

    mA0 = vdupq_n_f16(A[k + pc + (ic + ir + 2) * K]);
    mA1 = vdupq_n_f16(A[k + pc + (ic + ir + 3) * K]);
    result2_0 = vfmaq_f16(result2_0, mB0, mA0);
    result2_1 = vfmaq_f16(result2_1, mB1, mA0);
    result3_0 = vfmaq_f16(result3_0, mB0, mA1);
    result3_1 = vfmaq_f16(result3_1, mB1, mA1);

    mA0 = vdupq_n_f16(A[k + pc + (ic + ir + 4) * K]);
    mA1 = vdupq_n_f16(A[k + pc + (ic + ir + 5) * K]);
    result4_0 = vfmaq_f16(result4_0, mB0, mA0);
    result4_1 = vfmaq_f16(result4_1, mB1, mA0);
    result5_0 = vfmaq_f16(result5_0, mB0, mA1);
    result5_1 = vfmaq_f16(result5_1, mB1, mA1);
  }

  // Write registers back to C
  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result0_0))));
  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result0_1))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result1_0))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result1_1))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result2_0))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result2_1))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result3_0))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result3_1))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result4_0))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result4_1))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8]),
                      vcvt_f32_f16(vget_low_f16(result5_0))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8]),
                      vcvt_f32_f16(vget_low_f16(result5_1))));

  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result0_0))));
  vst1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 0) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result0_1))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result1_0))));
  vst1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 1) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result1_1))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result2_0))));
  vst1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result2_1))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 2) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result3_0))));
  vst1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 3) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result3_1))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result4_0))));
  vst1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 4) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result4_1))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 0 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result5_0))));
  vst1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8 + 4],
            vaddq_f32(vld1q_f32(&C32[(ic + ir + 5) * N + jc + jr + 1 * 8 + 4]),
                      vcvt_f32_f16(vget_high_f16(result5_1))));
}



#endif
