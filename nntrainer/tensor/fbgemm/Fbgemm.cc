/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define FBGEMM_EXPORTS
#include "./Fbgemm.h"
#include <cpuinfo.h>
#include <functional>
#include <stdexcept>
#include "./ExecuteKernel.h"

#ifdef FBGEMM_MEASURE_TIME_BREAKDOWN
double packing_time = 0.0;
double computing_time = 0.0;
double run_time = 0.0;
#endif



// bool fbgemmSupportedCPU() {
// #if defined(__x86_64__) || defined(__i386__) || \
//     (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
//   return (cpuinfo_initialize() && fbgemmHasAvx2Support());
// #else
//   return cpuinfo_initialize();
// #endif
// }


#define INSTANTIATE_ACC_T(PACK_A)   \
  INSTANTIATE_BASE(PACK_A, int32_t) \
  INSTANTIATE_BASE(PACK_A, int16_t)

#undef INSTANTIATE_ACC_T
#undef INSTANTIATE_BASE

#define INSTANTIATE_BASE(ACC_T, SPATIAL_DIM)                        \
  template void fbgemmPacked(                            \
      PackMatrix<                                                   \
          PackAWithIm2Col<uint8_t, ACC_T, SPATIAL_DIM>,             \
          uint8_t,                                                  \
          ACC_T>& packA,                                            \
      PackMatrix<PackBMatrix<int8_t, ACC_T>, int8_t, ACC_T>& packB, \
      int32_t* C,                                                   \
      int32_t* C_buffer,                                            \
      uint32_t ldc,                                                 \
      const memCopy<>& outProcess,                                  \
      int thread_id,                                                \
      int num_threads,                                              \
      const BlockingFactors* blocking_params);

#define INSTANTIATE_SPATIAL_DIM(ACC_T) \
  INSTANTIATE_BASE(ACC_T, 1)           \
  INSTANTIATE_BASE(ACC_T, 2)           \
  INSTANTIATE_BASE(ACC_T, 3)


#undef INSTANTIATE_SPATIAL_DIM
#undef INSTANTIATE_BASE


template void fbgemmPacked(
    PackMatrix<PackAMatrix<uint8_t, int16_t>, uint8_t, int16_t>& packA,
    PackMatrix<PackBMatrix<int8_t, int16_t>, int8_t, int16_t>& packB,
    int32_t* C,
    int32_t* C_buffer,
    uint32_t ldc,
    const DoNothing<int32_t, int32_t>& outProcess,
    int thread_id,
    int num_threads,
    const BlockingFactors* blocking_params);

