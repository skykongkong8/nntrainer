// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   uhgemm_common.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is common settings for uhgemm
 *
 */

#define N_BLOCKING (768)
#define K_BLOCKING (256)
#define M_BLOCKING (4096)
#define GEMM_UNROLLING_16 (16)
#define GEMM_UNROLLING_8 (8)
#define GEMM_UNROLLING_4 (4)
#define GEMM_UNROLLING_1 (1)
#define VL_FP16 (8)
#define VL_FP16_HALF (4)

// #define vcvt_u16_u32(a) vreinterpret_u16_f16(vcvt_f16_f32(vcvtq_f32_u32(a)))
// #define vcvt_u32_u16(a) vreinterpretq_u32_f32(vcvt_f32_f16(vcvt_f16_u16(a)))

#define vcvt_u16_u32(a) vmovn_u32(a)
#define vcvt_u32_u16(a) vmovl_u16(a)

