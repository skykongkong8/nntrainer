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
#include <unistd.h>
#include <fstream>
#include <filesystem>

bool is_cpu_fully_utilized(int cpu_id) {
    std::ifstream cur_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + "/cpufreq/scaling_cur_freq");
    std::ifstream max_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + "/cpufreq/cpuinfo_max_freq");

    int cur_freq = 0, max_freq = 0;
    if (cur_file >> cur_freq && max_file >> max_freq) {
        float utilization_ratio = float(cur_freq) / max_freq;
        std::cout << "CPU utilization_ratio : " << utilization_ratio << "\n";
        return utilization_ratio > 0.95f; // 95%+ means likely fully utilized
    }

    return false;
}

void print_cpu_frequency() {
    for (int i = 0; i < 8; ++i) { // Change 8 to match your number of cores
        is_cpu_fully_utilized(i);
        std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpufreq/scaling_cur_freq";
        std::ifstream freq_file(path);
        if (freq_file.is_open()) {
            int freq_khz;
            freq_file >> freq_khz;
            std::cout << "CPU" << i << " Frequency: " << freq_khz / 1000.0 << " MHz\n";
        }
    }
}

void print_cpu_temperature() {
    for (const auto& entry : std::filesystem::directory_iterator("/sys/class/thermal/")) {
        std::string temp_path = entry.path().string() + "/temp";
        std::ifstream temp_file(temp_path);
        if (temp_file.is_open()) {
            int temp_millic;
            temp_file >> temp_millic;
            std::cout << entry.path().filename() << " Temp: " << temp_millic / 1000.0 << " °C\n";
        }
    }
}

void print_memory_usage() {
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long pages_total, resident;
        statm >> pages_total >> resident;
        long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;
        std::cout << "Virtual Memory: " << pages_total * page_size_kb << " KB\n";
        std::cout << "Resident Set Size (RSS): " << resident * page_size_kb << " KB\n";
    }
}

template <typename T>
static inline std::vector<T>
generate_random_vector(size_t size, float min_val = -1.F, float max_val = 1.F) {
  std::random_device rd;
  std::mt19937 gen(42);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

template <typename T>
inline std::vector<T> generate_random_positive_vector(size_t size,
                                                      float min_val = 0.F,
                                                      float max_val = 0.5F) {
  std::random_device rd;
  std::mt19937 gen(42);
  // std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min_val, max_val);
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = static_cast<T>(dist(gen));
  }
  return vec;
}

template <typename T>
static inline std::vector<T> generate_homogeneous_vector(size_t size, T value) {
  std::vector<T> vec(size);
  for (auto &val : vec) {
    val = value;
  }
  return vec;
}

template <typename T> static inline void print_matrix(T *src, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << src[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T>
inline void print_matrix_partially(T *src, int M, int N, int partial_m = 5,
                                   int partial_n = 5, int partial_len = 5) {
  for (int k = 0; k < partial_len; ++k) {
    std::cout << src[partial_m * N + partial_n + k] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
inline void print_vector_partially(T *src, int init_idx = 5,
                                   int partial_len = 5) {
  for (int k = 0; k < partial_len; ++k) {
    std::cout << src[init_idx + k] << " ";
  }
  std::cout << std::endl;
}

template <typename T>
static inline void
print_matrix_partially_n(const std::string &name, const T *src, int M, int N,
                         int partial_m = 5, int partial_n = 5) {
  std::cout << name << ":" << std::endl;
  std::cout << "--------------------------" << std::endl;
  for (int i = 0; i < partial_m; ++i) {
    for (int j = 0; j < partial_n; ++j) {
      std::cout << src[i * N + j] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "--------------------------" << std::endl;
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
  // std::cout << "err_sum : " << err_sum << std::endl;
  return max_diff;
}

#define QK4_0 32
typedef struct {
  uint16_t d;            // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0_testonly;

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

typedef struct {
  float d;                 // delta
  int8_t qs[256];          // quants
  int16_t bsums[256 / 16]; // sum of quants in groups of 16
} block_q8_K_testonly;

/**
 * @brief
 */
struct block_q4_Kx8_testonly {
  int16_t d[8];       // super-block scale for quantized scales
  int16_t dmin[8];    // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

static inline void print_q4_k_block_partially(void *block) {
  block_q4_K_testonly *b = (block_q4_K_testonly *)block;
  std::cout << "d : " << b->d << std::endl;
  std::cout << "dmin : " << b->dmin << std::endl;
  // std::cout << "qs : ";
  // for (int i = 0; i < 256/2; ++i){
  //     uint8_t packed_val = b->qs[i];
  //     uint8_t val1 = packed_val & 0x0F;
  //     uint8_t val2 = (packed_val >> 4) & 0x0F;
  //     std::cout << (int)val1 << " " << (int)val2 << " ";
  // }
  std::cout << "qs 5~8 : ";
  for (int i = 5; i < 8; ++i) {
    uint8_t packed_val = b->qs[i];
    uint8_t val1 = packed_val & 0x0F;
    uint8_t val2 = (packed_val >> 4) & 0x0F;
    std::cout << (int)val1 << " " << (int)val2 << " ";
  }
  std::cout << std::endl;
}

TEST(nntrainer_cpu_backend_standalone, DISABLED_ele_add) {
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

TEST(nntrainer_cpu_backend_standalone, q4_K_quantization) {
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

static float compute_mse(const uint32_t M, const uint32_t N,
                         std::vector<float> &ref_dst, std::vector<float> &dst) {
  auto mean_squared_error =
    mse<float, float>(ref_dst.data(), dst.data(), M * N);
  auto cos_sim = cosine_similarity(ref_dst.data(), dst.data(), M * N);
  auto max_differ = find_max_diff(ref_dst.data(), dst.data(), M, N);

  auto sum = std::accumulate(dst.begin(), dst.end(), 0.0);
  auto sum_gt = std::accumulate(ref_dst.begin(), ref_dst.end(), 0.0);
  std::cout << "[INFO]            MSE: " << mean_squared_error
            << ", COS_SIM: " << cos_sim << ", MAX_DIFFER: " << max_differ
            << ", SUM: " << sum << ", SUM_GT: " << sum_gt << std::endl;
  return mean_squared_error;
}

float test_gemm_q4_0(const uint32_t M, const uint32_t K,
                            const uint32_t N, const float *weights,
                            const float *activations,
                            std::vector<float> &ref_dst) {
  // needed to initialize f16 tables
  nntrainer::init_backend();

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
  nntrainer::repack_q4_0_to_q4_0_8(q4_0_repacked_qWeight.data(),
                                   q4_0_offline_qWeight_ptr, q4_0_data_size, N,
                                   K);

  // Step3. Run GEMM! (Online activation quantization + kernel routine + return
  // float)
  sleep(1);
  const int TC = 1;
  double gflops = 2 * M * N * K;
  std::vector<float> dst(M * N);
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  for (int tc = 0; tc < TC; ++tc){
  nntrainer::gemm_q4_0(M, N, K, activations, K,
                       (void *)q4_0_repacked_qWeight.data(), N, dst.data(), N);
  }
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "[INFO] gemm_q4_0: " << dt.count() / TC << " ns " << " , gflops : " << 1e+9 * gflops / (dt.count() / TC)<< std::endl;

  // Step4. Compute quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst);
  return mean_squared_error;
}

float test_gemm_q4_K(const uint32_t M, const uint32_t K,
                            const uint32_t N, const float *weights,
                            const float *activations,
                            std::vector<float> &ref_dst) {
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
  sleep(1);
  const int TC = 1;
  double gflops = 2 * M * N * K;
  std::vector<float> dst(M * N);
  nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                    N, dst.data(), N);
  nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                    N, dst.data(), N);
  nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                    N, dst.data(), N);
  auto t1 = high_resolution_clock::now();
  // #### MAIN TESTED METHOD ####
  for (int tc = 0; tc < TC; ++tc){
    nntrainer::gemm_q4_K(M, N, K, activations, K, (void *)repacked_qWeight.data(),
                       N, dst.data(), N);
  }
  // #### MAIN TESTED METHOD ####
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "[INFO] gemm_q4_K: " << dt.count() / TC << " ns " << " , gflops : " << gflops * (1e+9) / (dt.count() / TC)<< std::endl;
  ///@note Needs validation!

  // Step4. Compare quantization error
  auto mean_squared_error = compute_mse(M, N, ref_dst, dst);
  return mean_squared_error;
}

static void run_quant_test(const uint32_t M, const uint32_t K, const uint32_t N,
                           float &q0_k_mse, float &q4_k_mse) {
  std::cout << "[INFO] Quantization Test (M:" << M << ", K:" << K << ", N:" << N
            << ")" << std::endl;
  ///@note A(M, K) * W.T(N, K) = (M, N)
  ///@note A(sizez, sizex) * W.T(sizey, sizex) = (sizez, sizey)

  ///@note q4_K GEMM is a Row-Major, transB GEMM
  // std::vector<float> activation = generate_homogeneous_vector<float>(M *
  // K, 2.0f); std::vector<float> weight = generate_homogeneous_vector<float>(N
  // * K, 1.0F);
  std::vector<float> activation = generate_random_vector<float>(M * K);
  std::vector<float> weight = generate_random_vector<float>(N * K);
  std::vector<float> ref_dst(M * N);

  // GROUND TRUTH TRANSB SGEMM for reference
const int TC = 1;
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc){
     nntrainer::sgemm(0, false, true, M, N, K, 1.F, activation.data(), K,
                   weight.data(), K, 0.F, ref_dst.data(), N);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "[INFO] sgemm : " << dt.count() / TC << " ns " << std::endl;

  q0_k_mse = test_gemm_q4_0(M, K, N, weight.data(), activation.data(), ref_dst);
  // q4_k_mse = test_gemm_q4_K(M, K, N, weight.data(), activation.data(), ref_dst);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512512128) {
  const unsigned int M = 512;
  const unsigned int K = 512;
  const unsigned int N = 128;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_128512512) {
  const unsigned int M = 128;
  const unsigned int K = 512;
  const unsigned int N = 512;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_256x1024x512) {
  const unsigned int M = 256;
  const unsigned int K = 1024;
  const unsigned int N = 512;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512x768x1024) {
  const unsigned int M = 512;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x1536x1536) {
  const unsigned int M = 1024;
  const unsigned int K = 1536;
  const unsigned int N = 1536;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x1536x5760) {
  const unsigned int M = 1024;
  const unsigned int K = 1536;
  const unsigned int N = 5760;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512x3072x512) {
  const unsigned int M = 512;
  const unsigned int K = 3072;
  const unsigned int N = 512;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_512x3072x3072) {
  const unsigned int M = 512;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x3072x8192) {
  const unsigned int M = 1024;
  const unsigned int K = 3072;
  const unsigned int N = 8192;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_1024x8192x3072) {
  const unsigned int M = 1024;
  const unsigned int K = 8192;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 2.0f);
  // ASSERT_LE(q4_k_mse, 2.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x512) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 512;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x1024) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 1024;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x1536x1536) {
  const unsigned int M = 1;
  const unsigned int K = 1536;
  const unsigned int N = 1536;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x1536x5760) {
  const unsigned int M = 1;
  const unsigned int K = 1536;
  const unsigned int N = 5760;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x3072) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x2048x8192) {
  const unsigned int M = 1;
  const unsigned int K = 2048;
  const unsigned int N = 8192;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x3072x8192) {
  const unsigned int M = 1;
  const unsigned int K = 3072;
  const unsigned int N = 8192;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x8192x3072) {
  const unsigned int M = 1;
  const unsigned int K = 8192;
  const unsigned int N = 3072;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMV_1x768x768) {
  const unsigned int M = 1;
  const unsigned int K = 768;
  const unsigned int N = 768;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

TEST(nntrainer_cpu_backend_standalone, quant_GEMM_2048x768x768) {
  const unsigned int M = 2048;
  const unsigned int K = 768;
  const unsigned int N = 768;
  float q0_k_mse, q4_k_mse;
  run_quant_test(M, K, N, q0_k_mse, q4_k_mse);
  ASSERT_LE(q0_k_mse, 1.0f);
  // ASSERT_LE(q4_k_mse, 1.0f);
}

#endif

/**
 * Room for optimization
 *
 *   1. Why don't we save weights for GEMM in q4_K_8x8 format offline?
 *        - PRO : We can save the time for repacking the weight
 *        - CON : We need to save the weight in two different formats (q4_K and
 *   q4_K_8x8), and such kernel works for specific HWs.
 *    2. Pre-allocation of runtime quantized activation buffer
 *   3. ???
 * 
 * 
 * [==========] Running 22 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 22 tests from nntrainer_cpu_backend_standalone
[ RUN      ] nntrainer_cpu_backend_standalone.q4_K_quantization
[       OK ] nntrainer_cpu_backend_standalone.q4_K_quantization (58 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_512512128
[INFO] Quantization Test (M:512, K:512, N:128)
[INFO] sgemm : 749205 ns 
[INFO] gemm_q4_0: 913758 ns  , gflops : 7.34427e+10
[INFO]            MSE: 0.246287, COS_SIM: 0.998931, MAX_DIFFER: 3.04886, SUM: 22889, SUM_GT: 23229.7
[INFO] gemm_q4_K: 999150 ns  , gflops : 6.7166e+10
[INFO]            MSE: 0.170702, COS_SIM: 0.999245, MAX_DIFFER: 1.84505, SUM: 23237.2, SUM_GT: 23229.7
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_512512128 (2032 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_128512512
[INFO] Quantization Test (M:128, K:512, N:512)
[INFO] sgemm : 11566249 ns 
[INFO] gemm_q4_0: 911006 ns  , gflops : 7.36646e+10
[INFO]            MSE: 0.245997, COS_SIM: 0.998931, MAX_DIFFER: 3.04886, SUM: 22963.8, SUM_GT: 23229.7
[INFO] gemm_q4_K: 905933 ns  , gflops : 7.40771e+10
[INFO]            MSE: 0.17046, COS_SIM: 0.999246, MAX_DIFFER: 1.84505, SUM: 23098.1, SUM_GT: 23229.7
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_128512512 (2050 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_256x1024x512
[INFO] Quantization Test (M:256, K:1024, N:512)
[INFO] sgemm : 5101205 ns 
[INFO] gemm_q4_0: 1450350 ns  , gflops : 1.85083e+11
[INFO]            MSE: 0.499131, COS_SIM: 0.999293, MAX_DIFFER: 4.55792, SUM: 82994.8, SUM_GT: 83756.7
[INFO] gemm_q4_K: 1332062 ns  , gflops : 2.01519e+11
[INFO]            MSE: 0.341125, COS_SIM: 0.9995, MAX_DIFFER: 3.00622, SUM: 83591.6, SUM_GT: 83756.7
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_256x1024x512 (2065 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_512x768x1024
[INFO] Quantization Test (M:512, K:768, N:1024)
[INFO] sgemm : 13144900 ns 
[INFO] gemm_q4_0: 2345686 ns  , gflops : 3.43314e+11
[INFO]            MSE: 0.365002, COS_SIM: 0.998794, MAX_DIFFER: 4.4057, SUM: 135409, SUM_GT: 137104
[INFO] gemm_q4_K: 5992615 ns  , gflops : 1.34383e+11
[INFO]            MSE: 0.255297, COS_SIM: 0.999144, MAX_DIFFER: 2.48805, SUM: 136570, SUM_GT: 137104
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_512x768x1024 (2121 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x1536x1536
[INFO] Quantization Test (M:1024, K:1536, N:1536)
[INFO] sgemm : 10652478 ns 
[INFO] gemm_q4_0: 12271055 ns  , gflops : 4.3751e+10
[INFO]            MSE: 0.734622, COS_SIM: 0.998943, MAX_DIFFER: 7.28125, SUM: 505216, SUM_GT: 512350
[INFO] gemm_q4_K: 15547793 ns  , gflops : 3.45304e+10
[INFO]            MSE: 0.511396, COS_SIM: 0.99925, MAX_DIFFER: 3.80975, SUM: 510213, SUM_GT: 512350
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x1536x1536 (2303 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x1536x5760
[INFO] Quantization Test (M:1024, K:1536, N:5760)
[INFO] sgemm : 27002108 ns 
[INFO] gemm_q4_0: 28636058 ns  , gflops : 3.28091e+10
[INFO]            MSE: 0.722979, COS_SIM: 0.998329, MAX_DIFFER: 7.28125, SUM: 514706, SUM_GT: 523890
[INFO] gemm_q4_K: 34597100 ns  , gflops : 2.71562e+10
[INFO]            MSE: 0.50736, COS_SIM: 0.998817, MAX_DIFFER: 3.92278, SUM: 521764, SUM_GT: 523890
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x1536x5760 (2916 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_512x3072x512
[INFO] Quantization Test (M:512, K:3072, N:512)
[INFO] sgemm : 3382267 ns 
[INFO] gemm_q4_0: 4665951 ns  , gflops : 3.45184e+11
[INFO]            MSE: 1.59499, COS_SIM: 0.999698, MAX_DIFFER: 12.0681, SUM: 518502, SUM_GT: 523401
[INFO] gemm_q4_K: 4600129 ns  , gflops : 3.50123e+11
[INFO]            MSE: 1.02895, COS_SIM: 0.999786, MAX_DIFFER: 5.05516, SUM: 522266, SUM_GT: 523401
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_512x3072x512 (2149 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_512x3072x3072
[INFO] Quantization Test (M:512, K:3072, N:3072)
[INFO] sgemm : 19098280 ns 
[INFO] gemm_q4_0: 18282959 ns  , gflops : 5.87291e+10
[INFO]            MSE: 1.47173, COS_SIM: 0.998941, MAX_DIFFER: 12.0681, SUM: 535578, SUM_GT: 541194
[INFO] gemm_q4_K: 21922024 ns  , gflops : 4.898e+10
[INFO]            MSE: 1.02035, COS_SIM: 0.999252, MAX_DIFFER: 5.45395, SUM: 540382, SUM_GT: 541194
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_512x3072x3072 (2777 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x3072x3072
[INFO] Quantization Test (M:1024, K:3072, N:3072)
[INFO] sgemm : 29052143 ns 
[INFO] gemm_q4_0: 31433258 ns  , gflops : 6.83188e+10
[INFO]            MSE: 1.47002, COS_SIM: 0.998941, MAX_DIFFER: 13.0698, SUM: 1.09108e+06, SUM_GT: 1.10098e+06
[INFO] gemm_q4_K: 32316225 ns  , gflops : 6.64522e+10
[INFO]            MSE: 1.01975, COS_SIM: 0.999251, MAX_DIFFER: 5.45395, SUM: 1.10019e+06, SUM_GT: 1.10098e+06
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x3072x3072 (2898 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x3072x8192
[INFO] Quantization Test (M:1024, K:3072, N:8192)
[INFO] sgemm : 77863107 ns 
[INFO] gemm_q4_0: 63200977 ns  , gflops : 0
[INFO]            MSE: 1.44029, COS_SIM: 0.99846, MAX_DIFFER: 13.0698, SUM: 1.12604e+06, SUM_GT: 1.13172e+06
[INFO] gemm_q4_K: 65444515 ns  , gflops : 0
[INFO]            MSE: 1.01194, COS_SIM: 0.99891, MAX_DIFFER: 5.5871, SUM: 1.13224e+06, SUM_GT: 1.13172e+06
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x3072x8192 (4176 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x8192x3072
[INFO] Quantization Test (M:1024, K:8192, N:3072)
[INFO] sgemm : 77373348 ns 
[INFO] gemm_q4_0: 58824876 ns  , gflops : 0
[INFO]            MSE: 4.02981, COS_SIM: 0.999422, MAX_DIFFER: 31.5134, SUM: 2.80158e+06, SUM_GT: 2.82824e+06
[INFO] gemm_q4_K: 68860479 ns  , gflops : 0
[INFO]            MSE: 2.73056, COS_SIM: 0.99959, MAX_DIFFER: 8.83098, SUM: 2.82361e+06, SUM_GT: 2.82824e+06
../test/unittest/unittest_nntrainer_cpu_backend.cpp:549: Failure
Expected: (q0_k_mse) <= (2.0f), actual: 4.02981 vs 2
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x8192x3072 (4130 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x768x512
[INFO] Quantization Test (M:1, K:768, N:512)
[INFO] sgemm : 230202 ns 
[INFO] gemm_q4_0: 612900 ns  , gflops : 1.28313e+09
[INFO]            MSE: 0.411046, COS_SIM: 0.999088, MAX_DIFFER: 2.78137, SUM: 334.875, SUM_GT: 316.769
[INFO] gemm_q4_K: 3230694 ns  , gflops : 2.43425e+08
[INFO]            MSE: 0.252386, COS_SIM: 0.999426, MAX_DIFFER: 1.38563, SUM: 320.268, SUM_GT: 316.769
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMV_1x768x512 (2046 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x768x1024
[INFO] Quantization Test (M:1, K:768, N:1024)
[INFO] sgemm : 437581 ns 
[INFO] gemm_q4_0: 624822 ns  , gflops : 2.5173e+09
[INFO]            MSE: 0.380304, COS_SIM: 0.998789, MAX_DIFFER: 2.78137, SUM: 457.617, SUM_GT: 425.798
[INFO] gemm_q4_K: 2511052 ns  , gflops : 6.26377e+08
[INFO]            MSE: 0.247899, COS_SIM: 0.999201, MAX_DIFFER: 1.58325, SUM: 421.655, SUM_GT: 425.798
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMV_1x768x1024 (2075 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x1536x1536
[INFO] Quantization Test (M:1, K:1536, N:1536)
[INFO] sgemm : 226589 ns 
[INFO] gemm_q4_0: 743491 ns  , gflops : 6.34654e+09
[INFO]            MSE: 0.757263, COS_SIM: 0.998892, MAX_DIFFER: 4.68085, SUM: 1322.76, SUM_GT: 1332.58
[INFO] gemm_q4_K: 2804138 ns  , gflops : 1.68272e+09
[INFO]            MSE: 0.533422, COS_SIM: 0.999201, MAX_DIFFER: 2.71771, SUM: 1322.38, SUM_GT: 1332.58
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMV_1x1536x1536 (2162 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x1536x5760
[INFO] Quantization Test (M:1, K:1536, N:5760)
[INFO] sgemm : 1687304 ns 
[INFO] gemm_q4_0: 734087 ns  , gflops : 2.41044e+10
[INFO]            MSE: 0.719129, COS_SIM: 0.998334, MAX_DIFFER: 4.68085, SUM: 1506.22, SUM_GT: 1666.41
[INFO] gemm_q4_K: 1055210 ns  , gflops : 1.67689e+10
[INFO]            MSE: 0.514595, COS_SIM: 0.998803, MAX_DIFFER: 2.81291, SUM: 1701.41, SUM_GT: 1666.41
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMV_1x1536x5760 (2526 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x3072x3072
[INFO] Quantization Test (M:1, K:3072, N:3072)
[INFO] sgemm : 1543604 ns 
[INFO] gemm_q4_0: 1129115 ns  , gflops : 1.67161e+10
[INFO]            MSE: 1.46648, COS_SIM: 0.998962, MAX_DIFFER: 9.30304, SUM: -230.873, SUM_GT: 21.0852
[INFO] gemm_q4_K: 677799 ns  , gflops : 2.78466e+10
[INFO]            MSE: 1.02101, COS_SIM: 0.999262, MAX_DIFFER: 3.88089, SUM: 13.8479, SUM_GT: 21.0852
../test/unittest/unittest_nntrainer_cpu_backend.cpp:599: Failure
Expected: (q0_k_mse) <= (1.0f), actual: 1.46648 vs 1
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMV_1x3072x3072 (2562 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x2048x8192
[INFO] Quantization Test (M:1, K:2048, N:8192)
[INFO] sgemm : 2777704 ns 
[INFO] gemm_q4_0: 736203 ns  , gflops : 4.55777e+10
[INFO]            MSE: 0.970947, COS_SIM: 0.998283, MAX_DIFFER: 5.50641, SUM: 579.256, SUM_GT: 799.764
[INFO] gemm_q4_K: 730137 ns  , gflops : 4.59564e+10
[INFO]            MSE: 0.685047, COS_SIM: 0.998779, MAX_DIFFER: 3.18282, SUM: 947.704, SUM_GT: 799.764
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMV_1x2048x8192 (2973 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x3072x8192
[INFO] Quantization Test (M:1, K:3072, N:8192)
[INFO] sgemm : 7041456 ns 
[INFO] gemm_q4_0: 1182850 ns  , gflops : 4.25512e+10
[INFO]            MSE: 1.4486, COS_SIM: 0.998491, MAX_DIFFER: 9.30304, SUM: 972.475, SUM_GT: 1255.69
[INFO] gemm_q4_K: 1211652 ns  , gflops : 4.15397e+10
[INFO]            MSE: 1.02578, COS_SIM: 0.998919, MAX_DIFFER: 4.33735, SUM: 1334.1, SUM_GT: 1255.69
../test/unittest/unittest_nntrainer_cpu_backend.cpp:619: Failure
Expected: (q0_k_mse) <= (1.0f), actual: 1.4486 vs 1
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMV_1x3072x8192 (3465 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x8192x3072
[INFO] Quantization Test (M:1, K:8192, N:3072)
[INFO] sgemm : 3504014 ns 
[INFO] gemm_q4_0: 817225 ns  , gflops : 6.15885e+10
[INFO]            MSE: 3.97812, COS_SIM: 0.999429, MAX_DIFFER: 23.5466, SUM: 2154.15, SUM_GT: 2237.81
[INFO] gemm_q4_K: 1198209 ns  , gflops : 4.20057e+10
[INFO]            MSE: 2.70243, COS_SIM: 0.999594, MAX_DIFFER: 7.30078, SUM: 2054.5, SUM_GT: 2237.81
../test/unittest/unittest_nntrainer_cpu_backend.cpp:629: Failure
Expected: (q0_k_mse) <= (1.0f), actual: 3.97812 vs 1
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMV_1x8192x3072 (3446 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMV_1x768x768
[INFO] Quantization Test (M:1, K:768, N:768)
[INFO] sgemm : 1949107 ns 
[INFO] gemm_q4_0: 1052006 ns  , gflops : 1.12133e+09
[INFO]            MSE: 0.392532, COS_SIM: 0.998912, MAX_DIFFER: 2.78137, SUM: 259.765, SUM_GT: 230.684
[INFO] gemm_q4_K: 661475 ns  , gflops : 1.78336e+09
[INFO]            MSE: 0.244852, COS_SIM: 0.99931, MAX_DIFFER: 1.58325, SUM: 232.308, SUM_GT: 230.684
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMV_1x768x768 (2058 ms)
[ RUN      ] nntrainer_cpu_backend_standalone.quant_GEMM_2048x768x768
[INFO] Quantization Test (M:2048, K:768, N:768)
[INFO] sgemm : 6917586 ns 
[INFO] gemm_q4_0: 5618630 ns  , gflops : 4.29984e+11
[INFO]            MSE: 0.3637, COS_SIM: 0.998464, MAX_DIFFER: 4.4057, SUM: 197357, SUM_GT: 199079
[INFO] gemm_q4_K: 7436052 ns  , gflops : 3.24893e+11
[INFO]            MSE: 0.255405, COS_SIM: 0.998911, MAX_DIFFER: 2.49747, SUM: 198489, SUM_GT: 199079
[       OK ] nntrainer_cpu_backend_standalone.quant_GEMM_2048x768x768 (2142 ms)
[----------] 22 tests from nntrainer_cpu_backend_standalone (55130 ms total)

[----------] Global test environment tear-down
[==========] 22 tests from 1 test suite ran. (55131 ms total)
[  PASSED  ] 18 tests.
[  FAILED  ] 4 tests, listed below:
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMM_1024x8192x3072
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMV_1x3072x3072
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMV_1x3072x8192
[  FAILED  ] nntrainer_cpu_backend_standalone.quant_GEMV_1x8192x3072

 4 FAILED TESTS
  YOU HAVE 1 DISABLED TEST


 */

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
