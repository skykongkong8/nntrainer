#include <arm_neon.h>
#include <iostream>

#define VL_FP16 8
#define VL_FP16_HALF 4
#define VL_FP16_DOUBLE 16
#define VL_FP16_TRIPLE 24
#define A(i, j) A[(i) + (j)*LDA]
#define B(i, j) B[(i) + (j)*LDB]
#define C(i, j) C[(i) + (j)*LDC]

#define KERNEL_4x4_HGEMM()                  \
  a = vmul_f16(valpha, vld1_f16(&A(i, k))); \
  b0 = vmov_n_f16(B(k, j));                 \
  b1 = vmov_n_f16(B(k, j + 1));             \
  b2 = vmov_n_f16(B(k, j + 2));             \
  b3 = vmov_n_f16(B(k, j + 3));             \
  c0 = vfma_f16(c0, b0, a);                 \
  c1 = vfma_f16(c1, b1, a);                 \
  c2 = vfma_f16(c2, b2, a);                 \
  c3 = vfma_f16(c3, b3, a);                 \
  k++;

#define KERNEL_4X1_HGEMM()                  \
  a = vmul_f16(valpha, vld1_f16(&A(i, k))); \
  b0 = vmov_n_f16(B(k, j));                 \
  c0 = vfma_f16(c0, b0, a);                 \
  k++;

#define KERNEL_8x4_HGEMM()                    \
  a = vmulq_f16(valpha, vld1q_f16(&A(i, k))); \
  b0 = vmovq_n_f16(B(k, j));                  \
  b1 = vmovq_n_f16(B(k, j + 1));              \
  b2 = vmovq_n_f16(B(k, j + 2));              \
  b3 = vmovq_n_f16(B(k, j + 3));              \
  c0 = vfmaq_f16(c0, b0, a);                  \
  c1 = vfmaq_f16(c1, b1, a);                  \
  c2 = vfmaq_f16(c2, b2, a);                  \
  c3 = vfmaq_f16(c3, b3, a);                  \
  k++;

#define KERNEL_16x4_HGEMM()                              \
  a0 = vmulq_f16(valpha, vld1q_f16(&A(i, k)));           \
  a1 = vmulq_f16(valpha, vld1q_f16(&A(i + VL_FP16, k))); \
  b0 = vmovq_n_f16(B(k, j));                             \
  b1 = vmovq_n_f16(B(k, j + 1));                         \
  b2 = vmovq_n_f16(B(k, j + 2));                         \
  b3 = vmovq_n_f16(B(k, j + 3));                         \
  c00 = vfmaq_f16(c00, b0, a0);                          \
  c01 = vfmaq_f16(c01, b0, a1);                          \
  c10 = vfmaq_f16(c10, b1, a0);                          \
  c11 = vfmaq_f16(c11, b1, a1);                          \
  c20 = vfmaq_f16(c20, b2, a0);                          \
  c21 = vfmaq_f16(c21, b2, a1);                          \
  c30 = vfmaq_f16(c30, b3, a0);                          \
  c31 = vfmaq_f16(c31, b3, a1);                          \
  k++;

#define KERNEL_8x4_HGEMM_packing()                 \
  a = vmulq_f16(valpha, vld1q_f16(ptr_packing_a)); \
  b0 = vmovq_n_f16(*ptr_packing_b);                \
  b1 = vmovq_n_f16(*(ptr_packing_b + 1));          \
  b2 = vmovq_n_f16(*(ptr_packing_b + 2));          \
  b3 = vmovq_n_f16(*(ptr_packing_b + 3));          \
  c0 = vfmaq_f16(c0, b0, a);                       \
  c1 = vfmaq_f16(c1, b1, a);                       \
  c2 = vfmaq_f16(c2, b2, a);                       \
  c3 = vfmaq_f16(c3, b3, a);                       \
  k++;                                             \
  ptr_packing_a += VL_FP16;                        \
  ptr_packing_b += 4;

#define macro_KERNEL_8xkx4_HGEMM_packing()                         \
  c0 = vdupq_n_f16(0.f);                                           \
  c1 = vdupq_n_f16(0.f);                                           \
  c2 = vdupq_n_f16(0.f);                                           \
  c3 = vdupq_n_f16(0.f);                                           \
  for (k = 0; k < K_;) {                                           \
    KERNEL_8x4_HGEMM_packing();                                    \
    KERNEL_8x4_HGEMM_packing();                                    \
    KERNEL_8x4_HGEMM_packing();                                    \
    KERNEL_8x4_HGEMM_packing();                                    \
  }                                                                \
  for (k = K_; k < K;) {                                           \
    KERNEL_8x4_HGEMM_packing();                                    \
  }                                                                \
  vst1q_f16(&C(i, j), vaddq_f16(c0, vld1q_f16(&C(i, j))));         \
  vst1q_f16(&C(i, j + 1), vaddq_f16(c1, vld1q_f16(&C(i, j + 1)))); \
  vst1q_f16(&C(i, j + 2), vaddq_f16(c2, vld1q_f16(&C(i, j + 2)))); \
  vst1q_f16(&C(i, j + 3), vaddq_f16(c3, vld1q_f16(&C(i, j + 3))));

#define KERNEL_16x4_HGEMM_packing()                           \
  a0 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a));           \
  a1 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a + VL_FP16)); \
  b0 = vmovq_n_f16(*ptr_packing_b);                           \
  b1 = vmovq_n_f16(*(ptr_packing_b + 1));                     \
  b2 = vmovq_n_f16(*(ptr_packing_b + 2));                     \
  b3 = vmovq_n_f16(*(ptr_packing_b + 3));                     \
  c00 = vfmaq_f16(c00, b0, a0);                               \
  c01 = vfmaq_f16(c01, b0, a1);                               \
  c10 = vfmaq_f16(c10, b1, a0);                               \
  c11 = vfmaq_f16(c11, b1, a1);                               \
  c20 = vfmaq_f16(c20, b2, a0);                               \
  c21 = vfmaq_f16(c21, b2, a1);                               \
  c30 = vfmaq_f16(c30, b3, a0);                               \
  c31 = vfmaq_f16(c31, b3, a1);                               \
  k++;                                                        \
  ptr_packing_a += VL_FP16_DOUBLE;                            \
  ptr_packing_b += 4;

#define macro_KERNEL_16xkx4_HGEMM_packing()                         \
  c00 = vdupq_n_f16(0.f);                                           \
  c01 = vdupq_n_f16(0.f);                                           \
  c10 = vdupq_n_f16(0.f);                                           \
  c11 = vdupq_n_f16(0.f);                                           \
  c20 = vdupq_n_f16(0.f);                                           \
  c21 = vdupq_n_f16(0.f);                                           \
  c30 = vdupq_n_f16(0.f);                                           \
  c31 = vdupq_n_f16(0.f);                                           \
  for (k = 0; k < K_;) {                                            \
    KERNEL_16x4_HGEMM_packing();                                    \
    KERNEL_16x4_HGEMM_packing();                                    \
    KERNEL_16x4_HGEMM_packing();                                    \
    KERNEL_16x4_HGEMM_packing();                                    \
  }                                                                 \
  for (k = K_; k < K;) {                                            \
    KERNEL_16x4_HGEMM_packing();                                    \
  }                                                                 \
  vst1q_f16(&C(i, j), vaddq_f16(c00, vld1q_f16(&C(i, j))));         \
  vst1q_f16(&C(i + VL_FP16, j),                                     \
            vaddq_f16(c01, vld1q_f16(&C(i + VL_FP16, j))));         \
  vst1q_f16(&C(i, j + 1), vaddq_f16(c10, vld1q_f16(&C(i, j + 1)))); \
  vst1q_f16(&C(i + VL_FP16, j + 1),                                 \
            vaddq_f16(c11, vld1q_f16(&C(i + VL_FP16, j + 1))));     \
  vst1q_f16(&C(i, j + 2), vaddq_f16(c20, vld1q_f16(&C(i, j + 2)))); \
  vst1q_f16(&C(i + VL_FP16, j + 2),                                 \
            vaddq_f16(c21, vld1q_f16(&C(i + VL_FP16, j + 2))));     \
  vst1q_f16(&C(i, j + 3), vaddq_f16(c30, vld1q_f16(&C(i, j + 3)))); \
  vst1q_f16(&C(i + VL_FP16, j + 3),                                 \
            vaddq_f16(c31, vld1q_f16(&C(i + VL_FP16, j + 3))));

#define KERNEL_8x4_HGEMM_packing_v2()               \
  a0 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a)); \
  b0 = vmovq_n_f16(*ptr_packing_b);                 \
  b1 = vmovq_n_f16(*(ptr_packing_b + 1));           \
  b2 = vmovq_n_f16(*(ptr_packing_b + 2));           \
  b3 = vmovq_n_f16(*(ptr_packing_b + 3));           \
  c00 = vfmaq_f16(c00, b0, a0);                     \
  c10 = vfmaq_f16(c10, b1, a0);                     \
  c20 = vfmaq_f16(c20, b2, a0);                     \
  c30 = vfmaq_f16(c30, b3, a0);                     \
  k++;                                              \
  ptr_packing_a += VL_FP16;                         \
  ptr_packing_b += 4;

#define macro_KERNEL_8xkx4_HGEMM_packing_v2()                       \
  c00 = vdupq_n_f16(0.f);                                           \
  c10 = vdupq_n_f16(0.f);                                           \
  c20 = vdupq_n_f16(0.f);                                           \
  c30 = vdupq_n_f16(0.f);                                           \
  for (k = 0; k < K_;) {                                            \
    KERNEL_8x4_HGEMM_packing_v2();                                  \
    KERNEL_8x4_HGEMM_packing_v2();                                  \
    KERNEL_8x4_HGEMM_packing_v2();                                  \
    KERNEL_8x4_HGEMM_packing_v2();                                  \
  }                                                                 \
  for (k = K_; k < K;) {                                            \
    KERNEL_8x4_HGEMM_packing_v2();                                  \
  }                                                                 \
  vst1q_f16(&C(i, j), vaddq_f16(c00, vld1q_f16(&C(i, j))));         \
  vst1q_f16(&C(i, j + 1), vaddq_f16(c10, vld1q_f16(&C(i, j + 1)))); \
  vst1q_f16(&C(i, j + 2), vaddq_f16(c20, vld1q_f16(&C(i, j + 2)))); \
  vst1q_f16(&C(i, j + 3), vaddq_f16(c30, vld1q_f16(&C(i, j + 3))));

#define KERNEL_24x8_HGEMM_packing()                                  \
  a0 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a));                  \
  a1 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a + VL_FP16));        \
  a2 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a + VL_FP16_DOUBLE)); \
  b0 = vmovq_n_f16(*ptr_packing_b);                                  \
  b1 = vmovq_n_f16(*(ptr_packing_b + 1));                            \
  c00 = vfmaq_f16(c00, b0, a0);                                      \
  c01 = vfmaq_f16(c01, b0, a1);                                      \
  c02 = vfmaq_f16(c02, b0, a2);                                      \
  c10 = vfmaq_f16(c10, b1, a0);                                      \
  c11 = vfmaq_f16(c11, b1, a1);                                      \
  c12 = vfmaq_f16(c12, b1, a2);                                      \
  b0 = vmovq_n_f16(*(ptr_packing_b + 2));                            \
  b1 = vmovq_n_f16(*(ptr_packing_b + 3));                            \
  c20 = vfmaq_f16(c20, b0, a0);                                      \
  c21 = vfmaq_f16(c21, b0, a1);                                      \
  c22 = vfmaq_f16(c22, b0, a2);                                      \
  c30 = vfmaq_f16(c30, b1, a0);                                      \
  c31 = vfmaq_f16(c31, b1, a1);                                      \
  c32 = vfmaq_f16(c32, b1, a2);                                      \
  b0 = vmovq_n_f16(*(ptr_packing_b + 4));                            \
  b1 = vmovq_n_f16(*(ptr_packing_b + 5));                            \
  c40 = vfmaq_f16(c40, b0, a0);                                      \
  c41 = vfmaq_f16(c41, b0, a1);                                      \
  c42 = vfmaq_f16(c42, b0, a2);                                      \
  c50 = vfmaq_f16(c50, b1, a0);                                      \
  c51 = vfmaq_f16(c51, b1, a1);                                      \
  c52 = vfmaq_f16(c52, b1, a2);                                      \
  b0 = vmovq_n_f16(*(ptr_packing_b + 6));                            \
  b1 = vmovq_n_f16(*(ptr_packing_b + 7));                            \
  c60 = vfmaq_f16(c60, b0, a0);                                      \
  c61 = vfmaq_f16(c61, b0, a1);                                      \
  c62 = vfmaq_f16(c62, b0, a2);                                      \
  c70 = vfmaq_f16(c70, b1, a0);                                      \
  c71 = vfmaq_f16(c71, b1, a1);                                      \
  c72 = vfmaq_f16(c72, b1, a2);                                      \
  ptr_packing_a += VL_FP16_TRIPLE;                                   \
  ptr_packing_b += VL_FP16;                                          \
  k++;

#define macro_KERNEL_24xkx8_packing()                                  \
  c00 = vdupq_n_f16(0.f);                                              \
  c01 = vdupq_n_f16(0.f);                                              \
  c02 = vdupq_n_f16(0.f);                                              \
  c10 = vdupq_n_f16(0.f);                                              \
  c11 = vdupq_n_f16(0.f);                                              \
  c12 = vdupq_n_f16(0.f);                                              \
  c20 = vdupq_n_f16(0.f);                                              \
  c21 = vdupq_n_f16(0.f);                                              \
  c22 = vdupq_n_f16(0.f);                                              \
  c30 = vdupq_n_f16(0.f);                                              \
  c31 = vdupq_n_f16(0.f);                                              \
  c32 = vdupq_n_f16(0.f);                                              \
  c40 = vdupq_n_f16(0.f);                                              \
  c41 = vdupq_n_f16(0.f);                                              \
  c42 = vdupq_n_f16(0.f);                                              \
  c50 = vdupq_n_f16(0.f);                                              \
  c51 = vdupq_n_f16(0.f);                                              \
  c52 = vdupq_n_f16(0.f);                                              \
  c60 = vdupq_n_f16(0.f);                                              \
  c61 = vdupq_n_f16(0.f);                                              \
  c62 = vdupq_n_f16(0.f);                                              \
  c70 = vdupq_n_f16(0.f);                                              \
  c71 = vdupq_n_f16(0.f);                                              \
  c72 = vdupq_n_f16(0.f);                                              \
  for (k = k_start; k < K_;) {                                         \
    KERNEL_24x8_HGEMM_packing();                                       \
    KERNEL_24x8_HGEMM_packing();                                       \
    KERNEL_24x8_HGEMM_packing();                                       \
    KERNEL_24x8_HGEMM_packing();                                       \
  }                                                                    \
  for (k = K_; k < k_end;) {                                           \
    KERNEL_24x8_HGEMM_packing();                                       \
  }                                                                    \
  vst1q_f16(&C(i, j), vaddq_f16(c00, vld1q_f16(&C(i, j))));            \
  vst1q_f16(&C(i + VL_FP16, j),                                        \
            vaddq_f16(c01, vld1q_f16(&C(i + VL_FP16, j))));            \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j),                                 \
            vaddq_f16(c02, vld1q_f16(&C(i + VL_FP16_DOUBLE, j))));     \
  vst1q_f16(&C(i, j + 1), vaddq_f16(c10, vld1q_f16(&C(i, j + 1))));    \
  vst1q_f16(&C(i + VL_FP16, j + 1),                                    \
            vaddq_f16(c11, vld1q_f16(&C(i + VL_FP16, j + 1))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 1),                             \
            vaddq_f16(c12, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 1)))); \
  vst1q_f16(&C(i, j + 2), vaddq_f16(c20, vld1q_f16(&C(i, j + 2))));    \
  vst1q_f16(&C(i + VL_FP16, j + 2),                                    \
            vaddq_f16(c21, vld1q_f16(&C(i + VL_FP16, j + 2))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 2),                             \
            vaddq_f16(c22, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 2)))); \
  vst1q_f16(&C(i, j + 3), vaddq_f16(c30, vld1q_f16(&C(i, j + 3))));    \
  vst1q_f16(&C(i + VL_FP16, j + 3),                                    \
            vaddq_f16(c31, vld1q_f16(&C(i + VL_FP16, j + 3))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 3),                             \
            vaddq_f16(c32, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 3)))); \
  vst1q_f16(&C(i, j + 4), vaddq_f16(c40, vld1q_f16(&C(i, j + 4))));    \
  vst1q_f16(&C(i + VL_FP16, j + 4),                                    \
            vaddq_f16(c41, vld1q_f16(&C(i + VL_FP16, j + 4))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 4),                             \
            vaddq_f16(c42, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 4)))); \
  vst1q_f16(&C(i, j + 5), vaddq_f16(c50, vld1q_f16(&C(i, j + 5))));    \
  vst1q_f16(&C(i + VL_FP16, j + 5),                                    \
            vaddq_f16(c51, vld1q_f16(&C(i + VL_FP16, j + 5))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 5),                             \
            vaddq_f16(c52, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 5)))); \
  vst1q_f16(&C(i, j + 6), vaddq_f16(c60, vld1q_f16(&C(i, j + 6))));    \
  vst1q_f16(&C(i + VL_FP16, j + 6),                                    \
            vaddq_f16(c61, vld1q_f16(&C(i + VL_FP16, j + 6))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 6),                             \
            vaddq_f16(c62, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 6)))); \
  vst1q_f16(&C(i, j + 7), vaddq_f16(c70, vld1q_f16(&C(i, j + 7))));    \
  vst1q_f16(&C(i + VL_FP16, j + 7),                                    \
            vaddq_f16(c71, vld1q_f16(&C(i + VL_FP16, j + 7))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 7),                             \
            vaddq_f16(c72, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 7))));

#define KERNEL_8x8_HGEMM_packing()                  \
  a0 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a)); \
  b0 = vmovq_n_f16(*ptr_packing_b);                 \
  b1 = vmovq_n_f16(*(ptr_packing_b + 1));           \
  c00 = vfmaq_f16(c00, b0, a0);                     \
  c10 = vfmaq_f16(c10, b1, a0);                     \
  b0 = vmovq_n_f16(*(ptr_packing_b + 2));           \
  b1 = vmovq_n_f16(*(ptr_packing_b + 3));           \
  c20 = vfmaq_f16(c20, b0, a0);                     \
  c30 = vfmaq_f16(c30, b1, a0);                     \
  b0 = vmovq_n_f16(*(ptr_packing_b + 4));           \
  b1 = vmovq_n_f16(*(ptr_packing_b + 5));           \
  c40 = vfmaq_f16(c40, b0, a0);                     \
  c50 = vfmaq_f16(c50, b1, a0);                     \
  b0 = vmovq_n_f16(*(ptr_packing_b + 6));           \
  b1 = vmovq_n_f16(*(ptr_packing_b + 7));           \
  c60 = vfmaq_f16(c60, b0, a0);                     \
  c70 = vfmaq_f16(c70, b1, a0);                     \
  ptr_packing_a += VL_FP16;                         \
  ptr_packing_b += VL_FP16;                         \
  k++;

#define macro_KERNEL_8xkx8_packing()                                \
  c00 = vdupq_n_f16(0.f);                                           \
  c10 = vdupq_n_f16(0.f);                                           \
  c20 = vdupq_n_f16(0.f);                                           \
  c30 = vdupq_n_f16(0.f);                                           \
  c40 = vdupq_n_f16(0.f);                                           \
  c50 = vdupq_n_f16(0.f);                                           \
  c60 = vdupq_n_f16(0.f);                                           \
  c70 = vdupq_n_f16(0.f);                                           \
  for (k = k_start; k < K_;) {                                      \
    KERNEL_8x8_HGEMM_packing();                                     \
    KERNEL_8x8_HGEMM_packing();                                     \
    KERNEL_8x8_HGEMM_packing();                                     \
    KERNEL_8x8_HGEMM_packing();                                     \
  }                                                                 \
  for (k = K_; k < k_end;) {                                        \
    KERNEL_8x8_HGEMM_packing();                                     \
  }                                                                 \
  vst1q_f16(&C(i, j), vaddq_f16(c00, vld1q_f16(&C(i, j))));         \
  vst1q_f16(&C(i, j + 1), vaddq_f16(c10, vld1q_f16(&C(i, j + 1)))); \
  vst1q_f16(&C(i, j + 2), vaddq_f16(c20, vld1q_f16(&C(i, j + 2)))); \
  vst1q_f16(&C(i, j + 3), vaddq_f16(c30, vld1q_f16(&C(i, j + 3)))); \
  vst1q_f16(&C(i, j + 4), vaddq_f16(c40, vld1q_f16(&C(i, j + 4)))); \
  vst1q_f16(&C(i, j + 5), vaddq_f16(c50, vld1q_f16(&C(i, j + 5)))); \
  vst1q_f16(&C(i, j + 6), vaddq_f16(c60, vld1q_f16(&C(i, j + 6)))); \
  vst1q_f16(&C(i, j + 7), vaddq_f16(c70, vld1q_f16(&C(i, j + 7))));

#define KERNEL_4x8_HGEMM_packing()                  \
  da0 = vmul_f16(dvalpha, vld1_f16(ptr_packing_a)); \
  db0 = vmov_n_f16(*ptr_packing_b);                 \
  db1 = vmov_n_f16(*(ptr_packing_b + 1));           \
  dc00 = vfma_f16(dc00, db0, da0);                  \
  dc10 = vfma_f16(dc10, db1, da0);                  \
  db0 = vmov_n_f16(*(ptr_packing_b + 2));           \
  db1 = vmov_n_f16(*(ptr_packing_b + 3));           \
  dc20 = vfma_f16(dc20, db0, da0);                  \
  dc30 = vfma_f16(dc30, db1, da0);                  \
  db0 = vmov_n_f16(*(ptr_packing_b + 4));           \
  db1 = vmov_n_f16(*(ptr_packing_b + 5));           \
  dc40 = vfma_f16(dc40, db0, da0);                  \
  dc50 = vfma_f16(dc50, db1, da0);                  \
  db0 = vmov_n_f16(*(ptr_packing_b + 6));           \
  db1 = vmov_n_f16(*(ptr_packing_b + 7));           \
  dc60 = vfma_f16(dc60, db0, da0);                  \
  dc70 = vfma_f16(dc70, db1, da0);                  \
  ptr_packing_a += VL_FP16_HALF;                    \
  ptr_packing_b += VL_FP16;                         \
  k++;

#define macro_KERNEL_4xkx8_packing()                              \
  dc00 = vdup_n_f16(0.f);                                         \
  dc10 = vdup_n_f16(0.f);                                         \
  dc20 = vdup_n_f16(0.f);                                         \
  dc30 = vdup_n_f16(0.f);                                         \
  dc40 = vdup_n_f16(0.f);                                         \
  dc50 = vdup_n_f16(0.f);                                         \
  dc60 = vdup_n_f16(0.f);                                         \
  dc70 = vdup_n_f16(0.f);                                         \
  for (k = k_start; k < K_;) {                                    \
    KERNEL_4x8_HGEMM_packing();                                   \
    KERNEL_4x8_HGEMM_packing();                                   \
    KERNEL_4x8_HGEMM_packing();                                   \
    KERNEL_4x8_HGEMM_packing();                                   \
  }                                                               \
  for (k = K_; k < k_end;) {                                      \
    KERNEL_4x8_HGEMM_packing();                                   \
  }                                                               \
  vst1_f16(&C(i, j), vadd_f16(dc00, vld1_f16(&C(i, j))));         \
  vst1_f16(&C(i, j + 1), vadd_f16(dc10, vld1_f16(&C(i, j + 1)))); \
  vst1_f16(&C(i, j + 2), vadd_f16(dc20, vld1_f16(&C(i, j + 2)))); \
  vst1_f16(&C(i, j + 3), vadd_f16(dc30, vld1_f16(&C(i, j + 3)))); \
  vst1_f16(&C(i, j + 4), vadd_f16(dc40, vld1_f16(&C(i, j + 4)))); \
  vst1_f16(&C(i, j + 5), vadd_f16(dc50, vld1_f16(&C(i, j + 5)))); \
  vst1_f16(&C(i, j + 6), vadd_f16(dc60, vld1_f16(&C(i, j + 6)))); \
  vst1_f16(&C(i, j + 7), vadd_f16(dc70, vld1_f16(&C(i, j + 7))));

#define macro_KERNEL_1xkx8_packing()                  \
  sc0 = sc1 = sc2 = sc3 = sc4 = sc5 = sc6 = sc7 = 0.; \
  for (k = k_start; k < k_end; k++) {                 \
    sa = alpha * (*ptr_packing_a);                    \
    sb0 = *(ptr_packing_b);                           \
    sb1 = *(ptr_packing_b + 1);                       \
    sb2 = *(ptr_packing_b + 2);                       \
    sb3 = *(ptr_packing_b + 3);                       \
    sb4 = *(ptr_packing_b + 4);                       \
    sb5 = *(ptr_packing_b + 5);                       \
    sb6 = *(ptr_packing_b + 6);                       \
    sb7 = *(ptr_packing_b + 7);                       \
    sc0 += sa * sb0;                                  \
    sc1 += sa * sb1;                                  \
    sc2 += sa * sb2;                                  \
    sc3 += sa * sb3;                                  \
    sc4 += sa * sb4;                                  \
    sc5 += sa * sb5;                                  \
    sc6 += sa * sb6;                                  \
    sc7 += sa * sb7;                                  \
    ptr_packing_a++;                                  \
    ptr_packing_b += 8;                               \
  }                                                   \
  C(i, j) += sc0;                                     \
  C(i, j + 1) += sc1;                                 \
  C(i, j + 2) += sc2;                                 \
  C(i, j + 3) += sc3;                                 \
  C(i, j + 4) += sc4;                                 \
  C(i, j + 5) += sc5;                                 \
  C(i, j + 6) += sc6;                                 \
  C(i, j + 7) += sc7;

#define macro_KERNEL_1xkx4_packing()  \
  sc0 = sc1 = sc2 = sc3 == 0.;        \
  for (k = k_start; k < k_end; k++) { \
    sa = alpha * (*ptr_packing_a);    \
    sb0 = *(ptr_packing_b);           \
    sb1 = *(ptr_packing_b + 1);       \
    sb2 = *(ptr_packing_b + 2);       \
    sb3 = *(ptr_packing_b + 3);       \
    sc0 += sa * sb0;                  \
    sc1 += sa * sb1;                  \
    sc2 += sa * sb2;                  \
    sc3 += sa * sb3;                  \
    ptr_packing_a++;                  \
    ptr_packing_b += 4;               \
  }                                   \
  C(i, j) += sc0;                     \
  C(i, j + 1) += sc1;                 \
  C(i, j + 2) += sc2;                 \
  C(i, j + 3) += sc3;

#define KERNEL_24x4_HGEMM_packing()                                  \
  a0 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a));                  \
  a1 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a + VL_FP16));        \
  a2 = vmulq_f16(valpha, vld1q_f16(ptr_packing_a + VL_FP16_DOUBLE)); \
  b0 = vmovq_n_f16(*ptr_packing_b);                                  \
  b1 = vmovq_n_f16(*(ptr_packing_b + 1));                            \
  c00 = vfmaq_f16(c00, b0, a0);                                      \
  c01 = vfmaq_f16(c01, b0, a1);                                      \
  c02 = vfmaq_f16(c02, b0, a2);                                      \
  c10 = vfmaq_f16(c10, b1, a0);                                      \
  c11 = vfmaq_f16(c11, b1, a1);                                      \
  c12 = vfmaq_f16(c12, b1, a2);                                      \
  b0 = vmovq_n_f16(*(ptr_packing_b + 2));                            \
  b1 = vmovq_n_f16(*(ptr_packing_b + 3));                            \
  c20 = vfmaq_f16(c20, b0, a0);                                      \
  c21 = vfmaq_f16(c21, b0, a1);                                      \
  c22 = vfmaq_f16(c22, b0, a2);                                      \
  c30 = vfmaq_f16(c30, b1, a0);                                      \
  c31 = vfmaq_f16(c31, b1, a1);                                      \
  c32 = vfmaq_f16(c32, b1, a2);                                      \
  ptr_packing_a += VL_FP16_TRIPLE;                                   \
  ptr_packing_b += VL_FP16_HALF;                                     \
  k++;

#define macro_KERNEL_24xkx4_packing()                                  \
  c00 = vdupq_n_f16(0.f);                                              \
  c01 = vdupq_n_f16(0.f);                                              \
  c02 = vdupq_n_f16(0.f);                                              \
  c10 = vdupq_n_f16(0.f);                                              \
  c11 = vdupq_n_f16(0.f);                                              \
  c12 = vdupq_n_f16(0.f);                                              \
  c20 = vdupq_n_f16(0.f);                                              \
  c21 = vdupq_n_f16(0.f);                                              \
  c22 = vdupq_n_f16(0.f);                                              \
  c30 = vdupq_n_f16(0.f);                                              \
  c31 = vdupq_n_f16(0.f);                                              \
  c32 = vdupq_n_f16(0.f);                                              \
  for (k = k_start; k < K_;) {                                         \
    KERNEL_24x4_HGEMM_packing();                                       \
    KERNEL_24x4_HGEMM_packing();                                       \
    KERNEL_24x4_HGEMM_packing();                                       \
    KERNEL_24x4_HGEMM_packing();                                       \
  }                                                                    \
  for (k = K_; k < k_end;) {                                           \
    KERNEL_24x4_HGEMM_packing();                                       \
  }                                                                    \
  vst1q_f16(&C(i, j), vaddq_f16(c00, vld1q_f16(&C(i, j))));            \
  vst1q_f16(&C(i + VL_FP16, j),                                        \
            vaddq_f16(c01, vld1q_f16(&C(i + VL_FP16, j))));            \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j),                                 \
            vaddq_f16(c02, vld1q_f16(&C(i + VL_FP16_DOUBLE, j))));     \
  vst1q_f16(&C(i, j + 1), vaddq_f16(c10, vld1q_f16(&C(i, j + 1))));    \
  vst1q_f16(&C(i + VL_FP16, j + 1),                                    \
            vaddq_f16(c11, vld1q_f16(&C(i + VL_FP16, j + 1))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 1),                             \
            vaddq_f16(c12, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 1)))); \
  vst1q_f16(&C(i, j + 2), vaddq_f16(c20, vld1q_f16(&C(i, j + 2))));    \
  vst1q_f16(&C(i + VL_FP16, j + 2),                                    \
            vaddq_f16(c21, vld1q_f16(&C(i + VL_FP16, j + 2))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 2),                             \
            vaddq_f16(c22, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 2)))); \
  vst1q_f16(&C(i, j + 3), vaddq_f16(c30, vld1q_f16(&C(i, j + 3))));    \
  vst1q_f16(&C(i + VL_FP16, j + 3),                                    \
            vaddq_f16(c31, vld1q_f16(&C(i + VL_FP16, j + 3))));        \
  vst1q_f16(&C(i + VL_FP16_DOUBLE, j + 3),                             \
            vaddq_f16(c32, vld1q_f16(&C(i + VL_FP16_DOUBLE, j + 3))));
