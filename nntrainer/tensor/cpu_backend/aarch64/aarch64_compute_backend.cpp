// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file aarch64_compute_backend.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Compute backend for aarch64
 *
 */
#include <aarch64_compute_backend.h>
#include <assert.h>
#include <cblas_interface.h>
#include <fallback_internal.h>
#include <neon_single.h>
#include <nntrainer_error.h>
#ifdef ENABLE_FP16
#include <neon_half.h>
#endif

#define ROW_MAJOR 0
#define COL_MAJOR 1

namespace nntrainer {

#ifdef ENABLE_FP16
void sscal(const unsigned int N, const float alpha, _FP16 *X,
           const unsigned int incX) {
  assert(incX > 0);

  if (incX == 1) {
    nntrainer::neon::hscal(N, X, alpha);
  } else {
    __fallback_sscal(N, alpha, X, incX);
  }
}

_FP16 snrm2(const unsigned int N, const _FP16 *X, const unsigned int incX) {
  assert(incX > 0);
  _FP16 sum;
  if (incX == 1) {
    sum = nntrainer::neon::hnrm2(N, X);
  } else {
    sum = __fallback_snrm2(N, X, incX);
  }
  return sum;
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::hcopy(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp32_to_fp16(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy(const unsigned int N, const _FP16 *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_fp16_to_fp32(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int4_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int4_to_float16(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float16(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, _FP16 *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp16(N, X, Y);
  } else {
    __fallback_scopy_int8_to_float16(N, X, incX, Y, incY);
  }
}

_FP16 sdot(const unsigned int N, const _FP16 *X, const unsigned int incX,
           const _FP16 *Y, const unsigned int incY) {
  _FP16 ret = 0;
  assert(incX > 0 && incY > 0);
  if (incX == 1 && incY == 1) {
    ret = nntrainer::neon::hdot(N, X, Y);
  } else {
    __fallback_sdot(N, X, incX, Y, incY);
  }
  return ret;
}

void saxpy(const unsigned int N, const float alpha, const _FP16 *X,
           const unsigned int incX, _FP16 *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::haxpy(N, alpha, X, Y);
  } else {
    __fallback_saxpy(N, alpha, X, incX, Y, incY);
  }
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const _FP16 *A, const unsigned int lda,
           const _FP16 *B, const unsigned int ldb, const float beta, _FP16 *C,
           const unsigned int ldc) {
  if (TStorageOrder) {
    __fallback_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                     ldb, beta, C, ldc);
  } else {
    nntrainer::neon::hgemm(A, B, C, M, N, K, alpha, beta, TransA, TransB);
  }
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const _FP16 *A,
           const unsigned int lda, const _FP16 *X, const unsigned int incX,
           const float beta, _FP16 *Y, const unsigned int incY) {
  if (TStorageOrder) {
    __fallback_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta,
                     Y, incY);
  } else {
    if (TransA) {
      nntrainer::neon::hgemv_transpose(A, X, Y, M, N, alpha, beta);
    } else {
      nntrainer::neon::hgemv(A, X, Y, M, N, alpha, beta);
    }
  }
}

void ele_mul(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_mul(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_add(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_sub(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const _FP16 *X, const _FP16 *Y, _FP16 *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_div(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

unsigned int isamax(const unsigned int N, const _FP16 *X,
                    const unsigned int incX) {
  unsigned int max_idx = 0;
  if (incX == 1 && N >= 8) {
    max_idx = nntrainer::neon::isamax(N, X);
  } else {
    max_idx = __fallback_isamax(N, X, incX);
  }
  return max_idx;
}

void inv_sqrt_inplace(const unsigned int N, _FP16 *X) {
  nntrainer::neon::inv_sqrt_inplace(N, X);
}
#endif
void scopy(const unsigned int N, const uint8_t *X, const unsigned int incX,
           uint8_t *Y, const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_or_int4(N, X, Y);
  } else {
    __fallback_scopy(N, X, incX, Y, incY);
  }
}

void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {
  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int4_to_fp32(N, X, Y);
  } else {
    __fallback_scopy_int4_to_float32(N, X, incX, Y, incY);
  }
}

void scopy_int8_to_float32(const unsigned int N, const uint8_t *X,
                           const unsigned int incX, float *Y,
                           const unsigned int incY) {

  if (incX == 1 && incY == 1) {
    nntrainer::neon::copy_int8_to_fp32(N, X, Y);
  } else {
    __fallback_scopy_int8_to_float32(N, X, incX, Y, incY);
  }
}

void sine(const unsigned int N, float *X, float *Y, float alpha) {
  nntrainer::neon::sine(N, X, Y, alpha);
}

void cosine(const unsigned int N, float *X, float *Y, float alpha) {
  nntrainer::neon::cosine(N, X, Y, alpha);
}

void inv_sqrt_inplace(const unsigned int N, float *X) {
  nntrainer::neon::inv_sqrt_inplace(N, X);
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_mul(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_mul(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_add(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_add(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_sub(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_sub(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_sub(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void ele_div(const unsigned N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (i_stride == 1 && o_stride == 1) {
    nntrainer::neon::ele_div(N, X, Y, Z, alpha, beta);
  } else
    __fallback_ele_div(N, X, Y, Z, alpha, beta, i_stride, o_stride);
}

void saxpy(const unsigned int N, const float alpha, const float *X,
           const unsigned int incX, float *Y, const unsigned int incY) {
  __cblas_saxpy(N, alpha, X, incX, Y, incY);
}

void sgemv(const unsigned int TStorageOrder, bool TransA, const unsigned int M,
           const unsigned int N, const float alpha, const float *A,
           const unsigned int lda, const float *X, const unsigned int incX,
           const float beta, float *Y, const unsigned int incY) {
  __cblas_sgemv(TStorageOrder, TransA, M, N, alpha, A, lda, X, incX, beta, Y,
                incY);
}

float sdot(const unsigned int N, const float *X, const unsigned int incX,
           const float *Y, const unsigned int incY) {
  return __cblas_sdot(N, X, incX, Y, incY);
}

void scopy(const unsigned int N, const float *X, const unsigned int incX,
           float *Y, const unsigned int incY) {
  __cblas_scopy(N, X, incX, Y, incY);
}

void sscal(const unsigned int N, const float alpha, float *X,
           const unsigned int incX) {
  __cblas_sscal(N, alpha, X, incX);
}

float snrm2(const unsigned int N, const float *X, const unsigned int incX) {
  return __cblas_snrm2(N, X, incX);
}

void sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
           const unsigned int M, const unsigned int N, const unsigned int K,
           const float alpha, const float *A, const unsigned int lda,
           const float *B, const unsigned int ldb, const float beta, float *C,
           const unsigned int ldc) {
  __cblas_sgemm(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                beta, C, ldc);
}

unsigned int isamax(const unsigned int N, const float *X,
                    const unsigned int incX) {
  return __cblas_isamax(N, X, incX);
}

} /* namespace nntrainer */