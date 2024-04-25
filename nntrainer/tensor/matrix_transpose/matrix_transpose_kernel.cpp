#include <arm_neon.h>
#include <matrix_transpose_kernel.h>

#define _TRANSPOSE4X4_FP32(r0, r1, r2, r3)                     \
  do {                                                       \
    float32x4_t tmp0, tmp1, tmp2, tmp3;                      \
    tmp0 = vzip1q_f32(r0, r1);                               \
    tmp2 = vzip1q_f32(r2, r3);                               \
    tmp1 = vzip2q_f32(r0, r1);                               \
    tmp3 = vzip2q_f32(r2, r3);                               \
    r0 = vcombine_f32(vget_low_f32(r0), vget_low_f32(r2));   \
    r1 = vcombine_f32(vget_high_f32(r2), vget_high_f32(r0)); \
    r2 = vcombine_f32(vget_low_f32(r1), vget_low_f32(r3));   \
    r3 = vcombine_f32(vget_high_f32(r3), vget_high_f32(r1)); \
  } while (0)

inline void transpose_kernel_4x4(const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst) {
  float32x4_t row0 = vld1q_f32(&src[0 * ld_src]);
  float32x4_t row1 = vld1q_f32(&src[1 * ld_src]);
  float32x4_t row2 = vld1q_f32(&src[2 * ld_src]);
  float32x4_t row3 = vld1q_f32(&src[3 * ld_src]);

  _TRANSPOSE4X4_FP32(row0, row1, row2, row3);

  vld1q_f32(&dst[0 * ld_dst], row0);
  vld1q_f32(&dst[1 * ld_dst], row1);
  vld1q_f32(&dst[2 * ld_dst], row2);
  vld1q_f32(&dst[3 * ld_dst], row3);
}
