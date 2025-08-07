// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file fallback_internal.cpp
 * @date   23 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback computation functions (raw implementation)
 *
 */

#include <algorithm>
#include <assert.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fallback_internal.h>
#include <stdexcept>
#include <tensor_dim.h>
#include <util_func.h>

#define sgemv_loop(ci, cj, cM, cN)                                             \
  do {                                                                         \
    float y0;                                                                  \
    unsigned int i, j;                                                         \
    for (ci = 0; ci != cM; ci++) {                                             \
      y0 = 0.0f;                                                               \
      if (beta != 0.0f) {                                                      \
        y0 = Y[ci * incY] * beta;                                              \
      }                                                                        \
      for (cj = 0; cj != cN; cj++)                                             \
        y0 += A[i + j * lda] * X[cj * incX];                                   \
      Y[ci * incY] = y0;                                                       \
    }                                                                          \
  } while (0);
namespace nntrainer {

void __fallback_int4_gemm(const int8_t *input, const uint8_t *qweight,
                          const float *scale, const int8_t *zero_point,
                          float *output, size_t B, size_t K, size_t N) {
  const size_t kpack = (K + 1) / 2;
  for (size_t b = 0; b < B; ++b) {
    for (size_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (size_t k = 0; k < K; ++k) {
        size_t packed_idx = n * kpack + k / 2;
        int8_t a = input[b * K + k];
        uint8_t packed = qweight[packed_idx];
        uint8_t nibble = (k % 2 == 0) ? (packed & 0x0F) : (packed >> 4 & 0x0F);
        int8_t w = (nibble > 7) ? (nibble - 16) : nibble;
        int8_t w_deq = w - zero_point[n];
        acc += a * w_deq;
      }
      output[b * N + n] = static_cast<float>(acc) * scale[n];
    }
  }
}

void int4_gemm_tiled(
    const int8_t* input,         // [B x K]
    const uint8_t* qweight,      // [N x ceil(K/2)] (4bit packed weights)
    const float* scale,          // [N]
    const int8_t* zero_point,    // [N]
    float* output,               // [B x N]
    size_t B, size_t K, size_t N,
    size_t tile_N = 8            // output channel tile size
) {
     constexpr size_t TILE_K = 128; // tile 크기 (K dimension 기준)

    for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            __m256i acc = _mm256_setzero_si256();

            for (size_t kt = 0; kt < K; kt += TILE_K) {
                size_t kend = std::min(K, kt + TILE_K);

                for (size_t k = kt; k + 32 <= kend; k += 32) {
                    const int8_t* a_ptr = input + b * K + k;

                    __m128i a_lo = _mm_loadu_si128((__m128i const*)(a_ptr));
                    __m128i a_hi = _mm_loadu_si128((__m128i const*)(a_ptr + 16));

                    const uint8_t* w_pack_ptr = qweight + n * (K / 2) + (k / 2);
                    alignas(32) int8_t w_deq[32];
                    for (int i = 0; i < 16; ++i) {
                        uint8_t packed = w_pack_ptr[i];
                        int8_t lo = static_cast<int8_t>(packed & 0x0F);
                        int8_t hi = static_cast<int8_t>((packed >> 4) & 0x0F);
                        w_deq[2 * i]     = (lo > 7) ? lo - 16 : lo;
                        w_deq[2 * i + 1] = (hi > 7) ? hi - 16 : hi;
                    }

                    __m128i w_lo = _mm_load_si128((__m128i const*)(w_deq));
                    __m128i w_hi = _mm_load_si128((__m128i const*)(w_deq + 16));

                    __m256i w_256 = _mm256_set_m128i(w_hi, w_lo);
                    __m256i zp = _mm256_set1_epi8(zero_point[n]);
                    w_256 = _mm256_sub_epi8(w_256, zp);

                    __m128i w_lo_zp = _mm256_castsi256_si128(w_256);
                    __m128i w_hi_zp = _mm256_extracti128_si256(w_256, 1);

                    __m256i w_lo_16 = _mm256_cvtepi8_epi16(w_lo_zp);
                    __m256i w_hi_16 = _mm256_cvtepi8_epi16(w_hi_zp);
                    __m256i a_lo_16 = _mm256_cvtepi8_epi16(a_lo);
                    __m256i a_hi_16 = _mm256_cvtepi8_epi16(a_hi);

                    __m256i mul_lo = _mm256_mullo_epi16(a_lo_16, w_lo_16);
                    __m256i mul_hi = _mm256_mullo_epi16(a_hi_16, w_hi_16);

                    __m256i sum32_lo = _mm256_madd_epi16(mul_lo, _mm256_set1_epi16(1));
                    __m256i sum32_hi = _mm256_madd_epi16(mul_hi, _mm256_set1_epi16(1));
                    __m256i sum32 = _mm256_add_epi32(sum32_lo, sum32_hi);

                    acc = _mm256_add_epi32(acc, sum32);
                }

                // Tail within tile
                for (size_t k = (kend & ~31); k < kend; ++k) {
                    int8_t a = input[b * K + k];
                    uint8_t packed = qweight[n * (K / 2) + k / 2];
                    uint8_t nibble = (k % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                    int8_t w = (nibble > 7) ? nibble - 16 : nibble;
                    int8_t w_deq = w - zero_point[n];
                    acc = _mm256_add_epi32(acc, _mm256_set1_epi32(a * w_deq)); // scalar broadcast
                }
            }

            // Store result
            alignas(32) int32_t acc_buf[8];
            _mm256_store_si256((__m256i*)acc_buf, acc);
            int32_t sum = 0;
            for (int i = 0; i < 8; ++i)
                sum += acc_buf[i];

            output[b * N + n] = scale[n] * static_cast<float>(sum);
        }
    }
}

void int4_gemm(
    const int8_t* input,         // [B x K]
    const uint8_t* qweight,      // [N x K/2] (4bit packed)
    const float* scale,          // [N]
    const int8_t* zero_point,    // [N]
    float* output,               // [B x N]
    size_t B, size_t K, size_t N // batch, in_features, out_features
) {

     for (size_t b = 0; b < B; ++b) {
        for (size_t n = 0; n < N; ++n) {
            __m256i acc = _mm256_setzero_si256(); 
            size_t k = 0;
            for (; k + 32 <= K; k += 32) {
                const int8_t* a_ptr = input + b * K + k;

                // --- Load input int8 (activation) ---
                __m128i a_lo = _mm_loadu_si128((__m128i const*)(a_ptr));        // [0..15]
                __m128i a_hi = _mm_loadu_si128((__m128i const*)(a_ptr + 16));   // [16..31]

                // --- Unpack 4bit → int8 weight vector (w_deq[32]) ---
                const uint8_t* w_pack_ptr = qweight + n * (K / 2) + (k / 2);
                alignas(32) int8_t w_deq[32];
                for (int i = 0; i < 16; ++i) {
                    uint8_t packed = w_pack_ptr[i];
                    int8_t lo = static_cast<int8_t>(packed & 0x0F);
                    int8_t hi = static_cast<int8_t>((packed >> 4) & 0x0F);
                    w_deq[2 * i]     = (lo > 7) ? lo - 16 : lo;
                    w_deq[2 * i + 1] = (hi > 7) ? hi - 16 : hi;
                }
                __m128i w_lo = _mm_load_si128((__m128i const*)(w_deq));         // [0..15]
                __m128i w_hi = _mm_load_si128((__m128i const*)(w_deq + 16));    // [16..31]

                // --- Zero point 처리 (fixed) ---
                __m256i w_256 = _mm256_set_m128i(w_hi, w_lo);
                __m256i zp = _mm256_set1_epi8(zero_point[n]);
                w_256 = _mm256_sub_epi8(w_256, zp);

                // 다시 나눠서 128-bit로 분리 후 변환
                __m128i w_lo_zp = _mm256_castsi256_si128(w_256);
                __m128i w_hi_zp = _mm256_extracti128_si256(w_256, 1);

                __m256i w_lo_16 = _mm256_cvtepi8_epi16(w_lo_zp);
                __m256i w_hi_16 = _mm256_cvtepi8_epi16(w_hi_zp);
                __m256i a_lo_16 = _mm256_cvtepi8_epi16(a_lo);
                __m256i a_hi_16 = _mm256_cvtepi8_epi16(a_hi);

                // int16 × int16 → int16 (pairwise)
                __m256i mul_lo = _mm256_mullo_epi16(a_lo_16, w_lo_16);
                __m256i mul_hi = _mm256_mullo_epi16(a_hi_16, w_hi_16);

                // int16 pair sum → int32
                __m256i sum32_lo = _mm256_madd_epi16(mul_lo, _mm256_set1_epi16(1));
                __m256i sum32_hi = _mm256_madd_epi16(mul_hi, _mm256_set1_epi16(1));
                __m256i sum32 = _mm256_add_epi32(sum32_lo, sum32_hi);

                acc = _mm256_add_epi32(acc, sum32);
            }

            // Tail 처리
            int32_t tail_acc = 0;
            for (; k < K; ++k) {
                int8_t a = input[b * K + k];
                uint8_t packed = qweight[n * (K / 2) + k / 2];
                uint8_t nibble = (k % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                int8_t w = (nibble > 7) ? nibble - 16 : nibble;
                int8_t w_deq = w - zero_point[n];
                tail_acc += static_cast<int32_t>(a) * static_cast<int32_t>(w_deq);
            }

            // Accumulate 8-lane int32 + tail
            alignas(32) int32_t acc_buf[8];
            _mm256_store_si256((__m256i*)acc_buf, acc);
            int32_t sum = tail_acc;
            for (int i = 0; i < 8; ++i)
                sum += acc_buf[i];

            output[b * N + n] = scale[n] * static_cast<float>(sum);
        }
    }
}


void __fallback_sscal(const unsigned int N, const float alpha, float *X,
                      const unsigned int incX) {
  assert(incX > 0);
  for (unsigned int i = 0; i < N; ++i)
    X[i * incX] = alpha * X[i * incX];
}

float __fallback_snrm2(const unsigned int N, const float *X,
                       const unsigned int incX) {
  assert(incX > 0);
  float sum = 0.0f;
  float tmp;

  for (unsigned int i = 0; i < N; i++) {
    tmp = X[i * incX];
    sum += tmp * tmp;
  }
  return sqrt(sum);
}

void __fallback_copy_s16_fp32(const unsigned int N, const int16_t *X,
                              float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_u16_fp32(const unsigned int N, const uint16_t *X,
                              float *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_u32(const unsigned int N, const float *X,
                              uint32_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_u16(const unsigned int N, const float *X,
                              uint16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_s16(const unsigned int N, const float *X,
                              int16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_s16(const unsigned int N, const int16_t *X, int16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_copy_u16(const unsigned int N, const uint16_t *X, uint16_t *Y) {
  for (unsigned int i = 0; i < N; ++i) {
    Y[i] = X[i];
  }
}

void __fallback_scopy(const unsigned int N, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = X[i * incX];
}

void __fallback_scopy(const unsigned int N, const uint8_t *X,
                      const unsigned int incX, uint8_t *Y,
                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy(const unsigned int N, const int8_t *X,
                      const unsigned int incX, int8_t *Y,
                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

/// @todo function with the same internal representation should be merged.
void __fallback_scopy_uint8_to_float32(const unsigned int N, const uint8_t *X,
                                       const unsigned int incX, float *Y,
                                       const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

void __fallback_scopy_int8_to_float32(const unsigned int N, const int8_t *X,
                                      const unsigned int incX, float *Y,
                                      const unsigned int incY) {
  for (unsigned int idx = 0; idx < N; idx++) {
    Y[idx * incX] = X[idx * incY];
  }
}

float __fallback_sdot(const unsigned int N, const float *X,
                      const unsigned int incX, const float *Y,
                      const unsigned int incY) {
  float ret = 0;
  for (unsigned int i = 0; i < N; ++i) {
    ret += X[i * incX] * Y[i * incY];
  }
  return ret;
}

void __fallback_saxpy(const unsigned int N, const float alpha, const float *X,
                      const unsigned int incX, float *Y,
                      const unsigned int incY) {
  assert(incX > 0 && incY > 0);
  for (unsigned int i = 0; i < N; ++i)
    Y[i * incY] = Y[i * incY] + X[i * incX] * alpha;
}

void __fallback_sgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const float *A,
                      const unsigned int lda, const float *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc) {
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      double c = 0.0;
      float c_old = C[m * ldc + n];
      for (unsigned int k = 0; k < K; ++k) {
        float a, b;
        a = ((TransA == true) ? A[k * lda + m] : A[m * lda + k]);
        b = ((TransB == true) ? B[n * ldb + k] : B[k * ldb + n]);
        c += a * b;
      }
      C[m * ldc + n] = alpha * c;
      if (beta != 0.0f) {
        C[m * ldc + n] += beta * c_old;
      }
    }
  }
}

void __fallback_sgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const float *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY) {

  if (TransA == true) {
    sgemv_loop(i, j, N, M);
  } else {
    sgemv_loop(j, i, M, N);
  }
}

unsigned int __fallback_isamax(const unsigned int N, const float *X,
                               const unsigned int incX) {
  unsigned int max_idx = 0;
  float max_val = X[0];
  for (unsigned int n = 1; n < N; n += incX) {
    float cur_val = std::abs(X[n]);
    if (cur_val > max_val) {
      max_val = cur_val;
      max_idx = n;
    }
  }

  return max_idx;
}

void __fallback_sine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::sin(alpha * X[i]);
    ++i;
  }
}

void __fallback_cosine(const unsigned int N, float *X, float *Y, float alpha) {
  unsigned int i = 0;
  while (i < N) {
    Y[i] = std::cos(alpha * X[i]);
    ++i;
  }
}

void __fallback_inv_sqrt_inplace(const unsigned int N, float *X) {
  for (unsigned int i = 0; i < N; ++i) {
    X[i] = 1 / std::sqrt(static_cast<float>(X[i]));
  }
}

void __fallback_ele_mul(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X * alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_add(const unsigned int N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X + alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_sub(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X - alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_ele_div(const unsigned N, const float *X, const float *Y,
                        float *Z, float alpha, float beta,
                        unsigned int i_stride, unsigned int o_stride) {
  for (unsigned int i = 0; i < N; ++i) {
    *Z = *X / (alpha * *Y) + ((0.0f == beta) ? 0.0f : beta * *Z);
    X += o_stride;
    Y += i_stride;
    Z += o_stride;
  }
}

void __fallback_transpose_matrix(const unsigned int M, const unsigned int N,
                                 const float *src, unsigned int ld_src,
                                 float *dst, unsigned int ld_dst) {
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

bool __fallback_isValid(const unsigned int N, const float *X) {
  for (size_t i = 0; i < N; ++i) {
    if (!isFloatValid(*X)) {
      return false;
    }
    ++X;
  }

  return true;
}

void __fallback_calc_trigonometric_vals_dup(unsigned int N_half, float *angle,
                                            float *cos_, float *sin_,
                                            unsigned int alpha) {
  throw std::runtime_error(
    "Error: No implementation of rotary embedding layer incremental_forwarding "
    "with SIMD acceleration except for NEON!");
}

void __fallback_swiglu(const unsigned int N, float *X, float *Y, float *Z) {
  unsigned int i = 0;
  while (i < N) {
    X[i] = (Y[i] / (1.f + std::exp(-Y[i]))) * Z[i];
    ++i;
  }
}

float __fallback_max(const unsigned int N, float *X) {
  std::vector<float> v(X, X + N);
  return *std::max_element(v.begin(), v.end());
}

void __fallback_softmax(const unsigned int N, float *X, float *Y) {
  unsigned int i = 0;
  float sum = 0.f;
  float max_x = __fallback_max(N, X);
  while (i < N) {
    sum += std::exp(X[i] - max_x);
    ++i;
  }
  i = 0;
  while (i < N) {
    Y[i] = std::exp(X[i] - max_x) / sum;
    ++i;
  }
}

void __fallback_gemm_q4_0(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q4_0");
}

void __fallback_gemm_q4_K(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q4_K");
}

float __fallback_dot_q6_K_q8_K(const unsigned int K, const void *v_q6_K,
                               const void *v_q8_K) {
  throw std::runtime_error("NYI : __fallback_dot_q6_K_q8_K");
  return 0;
}

float __fallback_dot_q6_K_f32(const unsigned int K, const void *v_q6_K,
                              const float *f) {
  throw std::runtime_error("NYI : __fallback_dot_q6_K_f32");
  return 0;
}

void __fallback_gemm_q6_K(const unsigned int M, const unsigned int N,
                          const unsigned int K, const float *A,
                          const unsigned int lda, const void *B,
                          const unsigned int ldb, float *C,
                          const unsigned int ldc) {
  throw std::runtime_error("NYI : __fallback_gemm_q6_K");
}

size_t __fallback_quantize_q4_0(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights) {
  throw std::runtime_error("NYI : __fallback_quantize_q4_0");
  return 1;
}

size_t __fallback_quantize_q4_K(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights) {
  throw std::runtime_error("NYI : __fallback_quantize_q4_K");
  return 1;
}

size_t __fallback_quantize_q6_K(const float *src, void *dst, int64_t nrow,
                                int64_t n_per_row, const float *quant_weights) {
  throw std::runtime_error("NYI : __fallback_quantize_q4_K");
  return 1;
}

void __fallback_dequantize_row_q4_K(const void *x_raw, float *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q4_K");
}

void __fallback_dequantize_row_q6_K(const void *x, float *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q6_K");
}

void __fallback_quantize_row_q6_K(const float *src, void *dst, int64_t k) {
  throw std::runtime_error("NYI : __fallback_quantize_row_q6_K");
}

void __fallback_quantize_row_q8_K(const float *src, void *dst, int64_t k) {
  throw std::runtime_error("NYI : __fallback_quantize_row_q8_K");
}

void __fallback_dequantize_row_q8_K(const void *x, float *y, int64_t k) {
  throw std::runtime_error("NYI : __fallback_dequantize_row_q8_K");
}

void __fallback_repack_q4_0_to_q4_0_8(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N) {
  throw std::runtime_error("NYI : __fallback_repack_q4_0_to_q4_0_8");
}

void __fallback_repack_q4_K_to_q4_K_8(void *W, void *repacked_W,
                                      size_t data_size, const unsigned int M,
                                      const unsigned int N) {
  throw std::runtime_error("NYI : __fallback_repack_q4_K_to_q4_K_8");
}

} // namespace nntrainer
