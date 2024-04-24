#include <matrix_transpose_kernel.h>
#include <arm_neon.h>

#define _TRANSPOSE4_FP32(r0, r1, r2, r3)\
do{\
    float32x4_t tmp0, tmp1, tmp2, tmp3;\
    tmp0 = vzip1q_f32(r0, r1);\
    tmp2 = vzip1q_f32(r2, r3);\
    tmp1 = vzip2q_f32(r0, r1);\
    tmp3 = vzip2q_f32(r2, r3);\

}while(0)\


inline void transpose_kernel_4x4(const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst){
                        float32x4_t             
                                 }
