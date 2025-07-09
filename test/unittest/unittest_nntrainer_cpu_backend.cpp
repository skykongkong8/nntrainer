// SPDX-License-Identifier: Apache-2.0
/**
 * @file	unittest_nntrainer_cpu_backend.cpp
 * @date	03 April 2025
 * @brief	This is unittest for cpu_backend standalone
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Sungsik Kong <ss.kong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "nntrainer_test_util.h"
#include <cpu_backend.h>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

#include <chrono>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::seconds;

template <typename T, bool random_init = false>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  auto init_val = random_init ? rd() : 42;
  std::mt19937 gen(init_val);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

template <typename T>
static inline double find_max_diff(T *src, T *src2, int M, int N) {
  float max_diff = 0;
  double err_sum = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      max_diff = std::max(max_diff, std::abs(src[i * N + j] - src2[i * N + j]));
      err_sum += std::abs(src[i * N + j] - src2[i * N + j]);
    }
  }
  std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

template<typename T>
void print_start_and_end_matrix(const unsigned int M, const unsigned int N, T* C){
  for (int i = 0; i < 3; ++i){
    std::cout << float(C[i]) << "\t";
  }
  std::cout << " ... ";
  for (int i = 0; i < 3; ++i){
    std::cout << float(C[(N)*(M-1) + i]) << "\t";
  }
  std::cout << std::endl;
}

#define QK4_0 32
#define QK8_0 32

/**
 * @brief q4_0 block
 *
 */
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;
/**
 * @brief q8_0 block
 * 
 */
typedef struct {
    uint16_t d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0_testonly;

/**
 * @brief q4_K block
 *
 */
typedef struct {
  union {
    struct {
      int16_t d;    // super-block scale for quantized scales
      int16_t dmin; // super-block scale for quantized mins
    };
    uint32_t dm;
  };
  uint8_t scales[12];  // scales and mins, quantized with 6 bits
  uint8_t qs[256 / 2]; // 4--bit quants
} block_q4_K_testonly;

/**
 * @brief q8_K block
 *
 */
typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;
/**
 * @brief q4_Kx8 block
 *
 */
struct block_q4_Kx8_testonly {
  int16_t d[8];       // super-block scale for quantized scales
  int16_t dmin[8];    // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

#define QK_K 256
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  uint16_t d;               // super-block scale
} block_q6_K_testonly;

/**
 * @brief Elementwise-addition unittest : Vanilla example for formulating a TC
 * in unittest_nntrainer_cpu_backend.cpp
 *
 */
TEST(nntrainer_cpu_backend_standalone, ele_add) {
  const unsigned int TEST_SIZE = 100;
  float alpha = 1.F;
  float beta = 0.F;
  unsigned int i_stride = 1;
  unsigned int o_stride = 1;

  std::vector<float> lhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> rhs = generate_random_vector<float>(TEST_SIZE);
  std::vector<float> dst(TEST_SIZE);

  const float *lhs_ptr = (const float *)lhs.data();
  const float *rhs_ptr = (const float *)rhs.data();
  float *dst_ptr = (float *)dst.data();

  nntrainer::ele_add(TEST_SIZE, lhs_ptr, rhs_ptr, dst_ptr, alpha, beta,
                     i_stride, o_stride);

  for (unsigned int i = 0; i < TEST_SIZE; ++i) {
    EXPECT_EQ(dst[i], lhs[i] + rhs[i]);
  }
}

#ifdef ENABLE_GGML
TEST(nntrainer_cpu_backend_standalone, q8_0_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q8_0_block_size = QK8_0;
  int64_t q8_0_type_size = sizeof(block_q8_0_testonly);
  int64_t num_blocks = (K * N) / q8_0_block_size;
  size_t data_size = q8_0_type_size * ne0 / q8_0_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q8_0(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q8_0(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  std::vector<_FP16> weight_f16 = generate_random_vector<_FP16>(N * K);
  std::vector<_FP16> weight_tmp_f16(N * K);
    const _FP16 *rhs_ptr_f16 = (const _FP16 *)weight_f16.data();
  _FP16 *rhs_ptr_tmp_f16 = weight_tmp_f16.data();

  nntrainer::quantize_q8_0(rhs_ptr_f16, (void *)offline_qWeight_ptr, K, N, nullptr);
  nntrainer::dequantize_row_q8_0(offline_qWeight_ptr, rhs_ptr_tmp_f16, K * N);

  const float eps = 1e-5;

  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);

  auto mean_squared_error_f16 =
    mse<_FP16, _FP16>(weight_f16.data(), rhs_ptr_tmp_f16, N * K);
  auto cos_sim_f16 = cosine_similarity(weight_f16.data(), rhs_ptr_tmp_f16, N * K);
  auto max_differ_f16 = find_max_diff(weight_f16.data(), rhs_ptr_tmp_f16, N, K);

  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error_f16, 0., eps * K * N);
  EXPECT_NEAR(cos_sim_f16, 0., eps * K * N);
  EXPECT_NEAR(max_differ_f16, 0., eps * K * N);

  std::cout << "mean_squared_error : " << mean_squared_error << " , mean_squared_error_f16 : " << mean_squared_error_f16 << std::endl;
  std::cout << "cos_sim : " << cos_sim << " , cos_sim_f16 : " << cos_sim_f16 << std::endl;
  std::cout << "max_differ : " << max_differ << " , max_differ_f16 : " << max_differ_f16 << std::endl;
  
  print_start_and_end_matrix<float>(N, K, rhs_ptr_tmp);
  print_start_and_end_matrix<_FP16>(N, K, rhs_ptr_tmp_f16);
}

TEST(nntrainer_cpu_backend_standalone, q8_0_quant_dequant_quant) {
  nntrainer::init_backend();

  const unsigned int K = 256;
  const unsigned int N = 256;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q8_0_block_size = QK8_0;
  int64_t q8_0_type_size = sizeof(block_q8_0_testonly);
  int64_t num_blocks = (K * N) / q8_0_block_size;
  size_t data_size = q8_0_type_size * ne0 / q8_0_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  std::vector<char> offline_qWeight2 = std::vector<char>(data_size);
  char *offline_qWeight_ptr2 = (char *)offline_qWeight2.data();

  nntrainer::quantize_q8_0(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q8_0(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  nntrainer::quantize_row_q8_0_ref_lossless(rhs_ptr_tmp, (void *)offline_qWeight_ptr2, K*N, offline_qWeight_ptr);
  
  for (int i = 0; i < num_blocks; ++i){
    auto first_block = ((block_q8_0_testonly *)((void *)offline_qWeight_ptr)) + i;
    auto second_block = ((block_q8_0_testonly *)((void *)offline_qWeight_ptr2)) + i;
    if (first_block->d != second_block->d) {
          std::cout << "Block1 " << i << " : d = " << first_block->d
                    <<std::endl;
          std::cout << "Block2 " << i << " : d = " << second_block->d
                    << std::endl;
    }
    for (int j = 0; j < QK8_0; ++j) {
      if (first_block->qs[j] != second_block->qs[j]) {
        std::cout << "Block1 " << i << " : qs[" << j << "] = "
                  << static_cast<int>(first_block->qs[j]) << std::endl;
        std::cout << "Block2 " << i << " : qs[" << j << "] = "
                  << static_cast<int>(second_block->qs[j]) << std::endl;
      }
    }
  }

}


TEST(nntrainer_cpu_backend_standalone, q4_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q4_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

TEST(nntrainer_cpu_backend_standalone, q4_K_quant_dequant_quant) {
  nntrainer::init_backend();

  const unsigned int K = 3072;
  const unsigned int N = 8192;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);
  std::vector<float> weight_tmp2(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();
  float *rhs_ptr_tmp2 = weight_tmp2.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * ne0 / q4_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  std::vector<char> offline_qWeight_2 = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();
  char *offline_qWeight_ptr_2 = (char *)offline_qWeight_2.data();

  nntrainer::quantize_q4_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  nntrainer::quantize_row_q4_K_ref_lossless(rhs_ptr_tmp, (void *)offline_qWeight_ptr_2, K*N, (void *)offline_qWeight_ptr);

  nntrainer::dequantize_row_q4_K(offline_qWeight_ptr_2, rhs_ptr_tmp2, K * N);

  if (true){
    for (int i = 0; i < num_blocks; ++i){
      auto first_block = ((block_q4_K_testonly *)((void *)offline_qWeight_ptr)) + i;
      auto second_block = ((block_q4_K_testonly *)((void *)offline_qWeight_ptr_2)) + i;
      if (first_block->d != second_block->d ||
          first_block->dmin != second_block->dmin) {
            std::cout << "Block1 " << i << " : d = " << first_block->d
                      << ", dmin = " << first_block->dmin << std::endl;
            std::cout << "Block2 " << i << " : d = " << second_block->d
                      << ", dmin = " << second_block->dmin << std::endl;
      }
      for (int j = 0; j < 12; ++j) {
        if (first_block->scales[j] != second_block->scales[j]) {
          std::cout << "Block1 " << i << " : scales[" << j << "] = "
                    << static_cast<int>(first_block->scales[j]) << std::endl;
          std::cout << "Block2 " << i << " : scales[" << j << "] = "
                    << static_cast<int>(second_block->scales[j]) << std::endl;
        }
      }
      for (int j = 0; j < 256 / 2; ++j) {
        if (first_block->qs[j] != second_block->qs[j]) {
          std::cout << "Block1 " << i << " : qs[" << j << "] = "
                    << static_cast<int>(first_block->qs[j]) << std::endl;
          std::cout << "Block2 " << i << " : qs[" << j << "] = "
                    << static_cast<int>(second_block->qs[j]) << std::endl;
        }
      }
    }
  }

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  auto mean_squared_error2 =
    mse<float, float>(weight.data(), rhs_ptr_tmp2, N * K);
  auto cos_sim2 = cosine_similarity(weight.data(), rhs_ptr_tmp2, N * K);
  auto max_differ2 = find_max_diff(weight.data(), rhs_ptr_tmp2, N, K);

  auto mean_squared_error3 =
    mse<float, float>(rhs_ptr_tmp, rhs_ptr_tmp2, N * K);
  auto cos_sim3 = cosine_similarity(rhs_ptr_tmp, rhs_ptr_tmp2, N * K);
  auto max_differ3 = find_max_diff(rhs_ptr_tmp, rhs_ptr_tmp2, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);

  std::cout << "mean_squared_error : " << mean_squared_error << " VS " << mean_squared_error2 << " VS " << mean_squared_error3 << std::endl;
  std::cout << "cos_sim : " << cos_sim << " VS " << cos_sim2 << " VS " << cos_sim3 << std::endl;
  std::cout << "max_differ : " << max_differ << " VS " << max_differ2  << " VS " << max_differ3 << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, q4_K_quant_dequant_quant_gemm) {
  nntrainer::init_backend();

  const unsigned int M = 3072;
  const unsigned int K = 3072;
  const unsigned int N = 8192;

  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();
  
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * N / q4_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  std::vector<char> offline_qWeight_2 = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();
  char *offline_qWeight_ptr_2 = (char *)offline_qWeight_2.data();

  nntrainer::quantize_q4_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q4_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  nntrainer::quantize_row_q4_K_ref_lossless(rhs_ptr_tmp, (void *)offline_qWeight_ptr_2, K* N, (void *)offline_qWeight_ptr);
  // nntrainer::quantize_q4_K(rhs_ptr_tmp, (void *)offline_qWeight_ptr_2, K, N, nullptr);

  if (true){
    for (int i = 0; i < num_blocks; ++i){
      auto first_block = ((block_q4_K_testonly *)((void *)offline_qWeight_ptr)) + i;
      auto second_block = ((block_q4_K_testonly *)((void *)offline_qWeight_ptr_2)) + i;
      if (first_block->d != second_block->d ||
          first_block->dmin != second_block->dmin) {
            std::cout << "Block1 " << i << " : d = " << first_block->d
                      << ", dmin = " << first_block->dmin << std::endl;
            std::cout << "Block2 " << i << " : d = " << second_block->d
                      << ", dmin = " << second_block->dmin << std::endl;
      }
      for (int j = 0; j < 12; ++j) {
        if (first_block->scales[j] != second_block->scales[j]) {
          std::cout << "Block1 " << i << " : scales[" << j << "] = "
                    << static_cast<int>(first_block->scales[j]) << std::endl;
          std::cout << "Block2 " << i << " : scales[" << j << "] = "
                    << static_cast<int>(second_block->scales[j]) << std::endl;
        }
      }
      for (int j = 0; j < 256 / 2; ++j) {
        if (first_block->qs[j] != second_block->qs[j]) {
          std::cout << "Block1 " << i << " : qs[" << j << "] = "
                    << static_cast<int>(first_block->qs[j]) << std::endl;
          std::cout << "Block2 " << i << " : qs[" << j << "] = "
                    << static_cast<int>(second_block->qs[j]) << std::endl;
        }
      }
    }
  }

  // Case#1 Ground Truth GEMM
  std::vector<char> repacked_qWeight = std::vector<char>(data_size);
  nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,
                                   data_size, N, K);
  std::vector<float> dst(M * N);
  nntrainer::gemm_q4_K(M, N, K, activation.data(), K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N);
  
  // Case#2 quant-dequant-quant GEMM -> How much loss occur?
  std::vector<char> repacked_qWeight2 = std::vector<char>(data_size);
  nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight2.data(), offline_qWeight_ptr_2,
                                   data_size, N, K);
  std::vector<float> dst2(M * N);
  nntrainer::gemm_q4_K(M, N, K, activation.data(), K, (void *)repacked_qWeight2.data(),
                      N, dst2.data(), N);

  // Case#3 quant-dequant-quant Values, but replace qparams with GT -> is this better than Case#2?
  // 1. Replace qparams from packed qWeight
  for (int nb = 0; nb < num_blocks/8; ++ nb){
    auto first_block = ((block_q4_Kx8_testonly *)((void *)repacked_qWeight.data())) + nb;
    auto second_block = ((block_q4_Kx8_testonly *)((void *)repacked_qWeight2.data())) + nb;
    for  (int j = 0; j < 8; ++j){
      second_block->d[j] = first_block->d[j];
      second_block->dmin[j] = first_block->dmin[j];
    }
    for (int j = 0; j < 96; ++j) {
      second_block->scales[j] = first_block->scales[j];
    }
    // for (int j = 0; j < 1024; ++j) { // with  uncommented this, MSE2 & MAX_DIFFER2 should go to ZERO!
    //   second_block->qs[j] = first_block->qs[j];
    // }
  }
  // 2. run GEMM with replaced qWeight!
  std::vector<float> dst3(M * N);
  nntrainer::gemm_q4_K(M, N, K, activation.data(), K, (void *)repacked_qWeight2.data(),
                      N, dst3.data(), N);

  // Case#4 quant-dequant-quant Values, but replace qparams with GT, but start from unpacked ones -> Verify if Case#3 is done well
  // 1. Replace qparams!
  for (int nb = 0; nb < num_blocks; ++nb){
    auto first_block = ((block_q4_K_testonly *)((void *)offline_qWeight_ptr)) + nb;
    auto second_block = ((block_q4_K_testonly *)((void *)offline_qWeight_ptr_2)) + nb;
    second_block->d = first_block->d;
    second_block->dmin = first_block->dmin;
    for (int j = 0; j < 12; ++j) {
      second_block->scales[j] = first_block->scales[j];
    }
    // for (int j = 0; j < 256 / 2; ++j) {
    //   second_block->qs[j] = first_block->qs[j];
    // }
  }
  // 2. Repack once again!
  nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight2.data(), offline_qWeight_ptr_2,
                                  data_size, N, K);
  // 3. run GEMM with replaced qWeight!
  std::vector<float> dst4(M * N);
  nntrainer::gemm_q4_K(M, N, K, activation.data(), K, (void *)repacked_qWeight2.data(),
                      N, dst4.data(), N);

  // GT VS quant-dequant
  auto mean_squared_error =
    mse<float, float>(dst.data(), dst2.data(),M*N);
  auto cos_sim = cosine_similarity(dst.data(), dst2.data(),M*N);
  auto max_differ = find_max_diff(dst.data(), dst2.data(), M, N);

  // GT VS replacing original qparams for packed-wise
  auto mean_squared_error2 =
    mse<float, float>(dst.data(), dst3.data(),M*N);
  auto cos_sim2 = cosine_similarity(dst.data(), dst3.data(),M*N);
  auto max_differ2 = find_max_diff(dst.data(), dst3.data(), M, N);

  // GT VS replacing original qparams for unpacked-wise
  auto mean_squared_error3 =
    mse<float, float>(dst.data(), dst4.data(),M*N);
  auto cos_sim3 = cosine_similarity(dst.data(), dst4.data(),M*N);
  auto max_differ3 = find_max_diff(dst.data(), dst4.data(), M, N);

  std::cout << "[INFO] MSE: " << mean_squared_error
            << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
            << std::endl;
  std::cout << "[INFO] MSE2: " << mean_squared_error2
            << ", COS_SIM2: " << cos_sim2 << ", MAX_DIFFER2: " << max_differ2
            << std::endl;
    std::cout << "[INFO] MSE3: " << mean_squared_error3
            << ", COS_SIM3: " << cos_sim3 << ", MAX_DIFFER3: " << max_differ3
            << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, q6_K_quantization) {
  nntrainer::init_backend();

  const unsigned int K = 768;
  const unsigned int N = 512;

  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> weight_tmp(N * K);

  const float *rhs_ptr = (const float *)weight.data();
  float *rhs_ptr_tmp = weight_tmp.data();

  int64_t ne0 = N; // row length of the weight matrix
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * ne0 / q6_k_block_size;
  data_size *= K;

  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  nntrainer::quantize_q6_K(rhs_ptr, (void *)offline_qWeight_ptr, K, N, nullptr);

  nntrainer::dequantize_row_q6_K(offline_qWeight_ptr, rhs_ptr_tmp, K * N);

  auto mean_squared_error =
    mse<float, float>(weight.data(), rhs_ptr_tmp, N * K);
  auto cos_sim = cosine_similarity(weight.data(), rhs_ptr_tmp, N * K);
  auto max_differ = find_max_diff(weight.data(), rhs_ptr_tmp, N, K);

  const float eps = 1e-5;
  ///@todo Find proper metric and standard to assess
  EXPECT_NEAR(mean_squared_error, 0., eps * K * N);
  EXPECT_NEAR(cos_sim, 0., eps * K * N);
  EXPECT_NEAR(max_differ, 0., eps * K * N);
}

float compute_mse(const uint32_t M, const uint32_t N,
                  std::vector<float> &ref_dst, std::vector<float> &dst,
                  bool print = false) {
  auto mean_squared_error =
    mse<float, float>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  if (print) {
    std::cout << "[INFO]            MSE: " << mean_squared_error
              << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
              << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  }
  return mean_squared_error;
}

float test_gemm_q4_0(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // needed to initialize f16 tables

  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_0_type_size = sizeof(block_q4_0_testonly);
  int64_t q4_0_block_size = 32;
  int64_t q4_0_num_blocks = (K * N) / q4_0_block_size;
  size_t q4_0_data_size = q4_0_type_size * N / q4_0_block_size;
  q4_0_data_size *= K;
  std::vector<char> q4_0_offline_qWeight = std::vector<char>(q4_0_data_size);

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  char *q4_0_offline_qWeight_ptr = (char *)q4_0_offline_qWeight.data();
  nntrainer::quantize_q4_0(weights, (void *)q4_0_offline_qWeight_ptr, N, K,
                           nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> q4_0_repacked_qWeight = std::vector<char>(q4_0_data_size);
  nntrainer::repack_q4_0(q4_0_repacked_qWeight.data(),
                                   q4_0_offline_qWeight_ptr, q4_0_data_size, N,
                                   K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_0: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  print_start_and_end_matrix(M, N, dst.data());

  return mean_squared_error;
}

float test_gemm_q4_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q4_k_block_size = 256;
  int64_t q4_k_type_size = sizeof(block_q4_K_testonly);
  int64_t num_blocks = (K * N) / q4_k_block_size;
  size_t data_size = q4_k_type_size * N / q4_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q4_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Repack Weight to q4_K_8x8 layout (This happens when you load the
  // model weights. It's a one-time operation)
  std::vector<char> repacked_qWeight = std::vector<char>(data_size);
  nntrainer::repack_q4_K_to_q4_K_8(repacked_qWeight.data(), offline_qWeight_ptr,
                                   data_size, N, K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q4_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  print_start_and_end_matrix(M, N, dst.data());

  return mean_squared_error;
}

float test_gemm_q6_K(const uint32_t M, const uint32_t K, const uint32_t N,
                     const float *weights, const float *activations,
                     std::vector<float> &ref_dst, bool print = false) {
  // Step0. Allocate a temporary buffer for quantized weight
  int64_t q6_k_block_size = 256;
  int64_t q6_k_type_size = sizeof(block_q6_K_testonly);
  int64_t num_blocks = (K * N) / q6_k_block_size;
  size_t data_size = q6_k_type_size * N / q6_k_block_size;
  data_size *= K;
  std::vector<char> offline_qWeight = std::vector<char>(data_size);
  char *offline_qWeight_ptr = (char *)offline_qWeight.data();

  // Step1. Supposed to be an offline Weight quantization from float to q4_K
  // (Zero latency overhead for the model runtime)
  nntrainer::quantize_q6_K(weights, (void *)offline_qWeight_ptr, N, K, nullptr);

  // Step2. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  std::vector<float> dst(M * N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  nntrainer::gemm_q6_K(M, N, K, activations, K, (void *)offline_qWeight_ptr, N,
                       dst.data(), N);
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] gemm_q6_K: " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst, print);

  print_start_and_end_matrix(M, N, dst.data());

  return mean_squared_error;
}

static void run_quant_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &q4_0_mse, float &q4_k_mse, float &q6_k_mse,
                           bool print = false) {
  nntrainer::init_backend();

  if (print) {
    std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K
              << ", N:" << N << ")" << std::endl;
  }
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);

  // GROUND TRUTH TRANSB SGEMM for reference
  auto t1 = high_resolution_clock::now();
  nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  if (print) {
    std::cout << "[INFO] sgemm :    " << dt.count() << " ns "
              << dt.count() / 1'000 << " us " << dt.count() / 1'000'000
              << " ms " << std::endl;
  }
  q4_0_mse =
    test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst,
    print);
  q4_k_mse =
    test_gemm_q4_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
  q6_k_mse =
    test_gemm_q6_K(M, K, N, weight.data(), activation.data(), ref_dst, print);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x1024x512) {
  const unsigned int M = 256;
  const unsigned int K = 1024;
  const unsigned int N = 512;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  // ASSERT_LE(q4_0_mse, 2.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_457x3072x3072) {
  const unsigned int M = 457;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  // ASSERT_LE(q4_0_mse, 1.5f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_458x3072x3072) {
  const unsigned int M = 458;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  // ASSERT_LE(q4_0_mse, 1.5f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_459x3072x3072) {
  const unsigned int M = 459;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  // ASSERT_LE(q4_0_mse, 1.5f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  ASSERT_LE(q4_0_mse, 2.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_3072x3072x8192) {
  const unsigned int M = 3072;
  const unsigned int K = 3072;
  const unsigned int N = 8192;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  ASSERT_LE(q4_0_mse, 2.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_3072x8192x3072) {
  const unsigned int M = 3072;
  const unsigned int K = 8192;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  ASSERT_LE(q4_0_mse, 2.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x3072) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q4_0_mse, q4_k_mse, q6_k_mse;
  constexpr float eps = 1e-5;
  run_quant_test(M, K, N, q4_0_mse, q4_k_mse, q6_k_mse, true);
  // ASSERT_LE(q4_0_mse, 1.0f);
  ASSERT_LE(q4_k_mse, eps * M * K * N);
  ASSERT_LE(q6_k_mse, q4_k_mse);
}


static void run_vec_dot_test(const uint32_t K, bool print = false) {
  const int TEST_CNT = 20;
  nanoseconds ref_time = (nanoseconds)0;
  nanoseconds q6_k_time = (nanoseconds)0;
  float ref_result, result;

  for (int i = -1; i < TEST_CNT; i++) {
    std::vector<float> activation = generate_random_vector<float, false>(K);
    std::vector<float> weight = generate_random_vector<float, false>(K);

    {
      // GROUND TRUTH sdot for reference
      auto t1 = high_resolution_clock::now();
      ref_result = nntrainer::sdot(K, weight.data(), 1, activation.data(), 1);
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        ref_time += dt;
      }
    }

    // Quantization of weights
    int64_t num_blocks = K / 256;
    size_t q6_k_data_size = num_blocks * sizeof(block_q6_K_testonly);
    std::vector<char> q6_K_weight = std::vector<char>(q6_k_data_size);
    nntrainer::quantize_row_q6_K(weight.data(), q6_K_weight.data(), K);

    // Quantization of activations
    int blocks_per_row = (K + QK_K - 1) / QK_K;
    int q8_K_activation_size = sizeof(block_q8_K_testonly) * blocks_per_row;
    std::vector<char> v_q8_activation = std::vector<char>(q8_K_activation_size);
    nntrainer::quantize_row_q8_K(activation.data(), v_q8_activation.data(), K);

    {
      auto t1 = high_resolution_clock::now();
      // #### MAIN TESTED METHOD ####
      result =
        nntrainer::dot_q6_K_q8_K(K, q6_K_weight.data(), v_q8_activation.data());
      // #### MAIN TESTED METHOD ####
      auto t2 = high_resolution_clock::now();
      auto dt = duration_cast<nanoseconds>(t2 - t1);
      if (i >= 0) { // skip the first run
        q6_k_time += dt;
      }
    }
    EXPECT_NEAR(result, ref_result, 0.25 * K / 256);
  }
  if (print) {
    std::cout << "[INFO] dot_q6_K_q8_K: TEST CNT: " << TEST_CNT << ", K: " << K
              << ", Average ref_time: " << ref_time.count() / TEST_CNT
              << " ns, Average q6_k_time: " << q6_k_time.count() / TEST_CNT
              << " ns " << std::endl;
  }
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_1024) {
  const uint32_t K = 1024;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_2560) {
  const uint32_t K = 2560;
  run_vec_dot_test(K);
}

TEST(nntrainer_cpu_backend_standalone, quant_q_6_K_DOT_10240) {
  const uint32_t K = 10240;
  run_vec_dot_test(K);
}

#endif

int main(int argc, char **argv) {
  int result = -1;
#ifdef ENABLE_GGML
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
#else
  result = 0;
#endif
  return result;
}
