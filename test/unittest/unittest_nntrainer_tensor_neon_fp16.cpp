// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file        unittest_nntrainer_tensor_neon_fp16.cpp
 * @date        03 August 2023
 * @brief       Unit test utility for tensor with NEON __fp16 support for ARM.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Debadri Samaddar <s.debadri@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds; // or micromicroseconds
#include <blas_neon.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

static void hgemm_raw_noTrans(const unsigned int M,
                      const unsigned int N, const unsigned int K,
                      const __fp16* A,
                      const __fp16* B,
                      __fp16* C) {

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      __fp16 c = 0.0;
      for (unsigned int k = 0; k < K; ++k) {
        __fp16 a, b;
        a = A[m * K + k];
        b =B[k * N + n];
        c += a * b;
      }
      C[m * N + n] = c;
    }
  }
}

TEST(nntrainer_Tensor, add_i) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // NEON fp16
  int result = input.add_i(input);

  // fp32
  result = input_fp32.add_i(input_fp32);

  float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
                                   input_fp32.getData<float>(), input.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  // conditions for fp16 sdot call:
  // this->(batch * channel * height) = arg->(width) = 1;

  size_t width = 23;

  __fp16 a_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                     12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);
  __fp16 b_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                     12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp16), b_data);

  float a_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                         12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);
  float b_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
                         12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  nntrainer::Tensor input_fp32_2(
    nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp32), b_data_fp32);

  nntrainer::Tensor result_neon;
  nntrainer::Tensor result_fp32;

  // NEON fp16
  result_neon = input.dot(input_2, false, false);

  // fp32
  result_fp32 = input_fp32.dot(input_fp32_2, false, false);

  float mseErrorNeon =
    mse<__fp16>(result_neon.getData<__fp16>(), result_fp32.getData<float>(),
                result_neon.size());

  double cosSimNeon =
    cosine_similarity<__fp16>(result_neon.getData<__fp16>(),
                              result_fp32.getData<float>(), result_neon.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, l2norm) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t width = 23;

  __fp16 a_data[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6, 0, 2.7,
                     2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

  float a_data_fp32[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6, 0, 2.7,
                         2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

  __fp16 result_neon;
  float result_fp32;

  // NEON fp16
  result_neon = input.l2norm();

  // fp32
  result_fp32 = input_fp32.l2norm();

  // absolute error
  const float epsilon = 1e-2;

  EXPECT_NEAR(result_neon, result_fp32, epsilon);
}

TEST(nntrainer_Tensor, multiply_i) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  // NEON fp16
  int result = input.multiply_i(0.1);

  // fp32
  result = input_fp32.multiply_i(0.1);

  float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
                                   input_fp32.getData<float>(), input.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, copy) {
  int batch = 1;
  int channel = 1;
  int height = 2;
  int width = 11;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-5;
  const float epsilon = 1e-4;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor output;
  nntrainer::Tensor output_fp32;

  // NEON fp16
  output.copy(input);

  // fp32
  output_fp32.copy(input_fp32);

  float mseErrorNeon = mse<__fp16>(output.getData<__fp16>(),
                                   output_fp32.getData<float>(), output.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    output.getData<__fp16>(), output_fp32.getData<float>(), output.size());

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, max_abs) {

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  size_t width = 25;

  __fp16 a_data[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
                     1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
                     2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
  nntrainer::Tensor input(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

  float a_data_fp32[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
                         1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
                         2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
  nntrainer::Tensor input_fp32(
    nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

  __fp16 result_neon;
  float result_fp32;

  // NEON fp16
  result_neon = input.max_abs();

  // fp32
  result_fp32 = input_fp32.max_abs();

  // absolute error
  const float epsilon = 1e-2;

  EXPECT_NEAR(result_neon, result_fp32, epsilon);
}

TEST(nntrainer_Tensor, sum_gemv_transpose_2_10) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
                          j * (batch * height) * alpha + k * (width)*alpha + l +
                          1);
  GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
                               j * (batch * height) * alpha +
                               k * (width)*alpha + l + 1);

  nntrainer::Tensor result0 = input.sum(0);
  nntrainer::Tensor result0_fp32 = input_fp32.sum(0);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, sum_gemv_2_10) {
  int batch = 3;
  int channel = 2;
  int height = 2;
  int width = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);

  const float alpha = 1e-1;
  const int MOD = 10;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);

  nntrainer::Tensor result0 = input.sum_by_batch();
  nntrainer::Tensor result0_fp32 = input_fp32.sum_by_batch();

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-4;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_16_8_16) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 16;
  int width = 8;

  int height_b = 8;
  int width_b = 16;

  bool transA = false;
  bool transB = false;

  const float alpha = 1e-1;
  const int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);

  nntrainer::Tensor C = A.dot(B, transA, transB);
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  C.print(std::cout);
  C_fp32.print(std::cout);

  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

// TEST(nntrainer_Tensor, dot_gemm_56_768_96000) {
//   /// @note GEMM : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 56;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 96000;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1) %
//                             MOD) *
//                              alpha);
//   auto t1 = high_resolution_clock::now();
//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   auto t2 = high_resolution_clock::now();
//   auto dt = duration_cast<microseconds>(t2 - t1);
//   std::cout << "function_to_assess : " << dt.count() << " microseconds "
//             << std::endl;
//   std::cout << "function_to_assess : "
//             << (height * width * height_b) / dt.count() << " Gflops "
//             << std::endl;

//   auto t3 = high_resolution_clock::now();
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
//   auto t4 = high_resolution_clock::now();
//   dt = duration_cast<microseconds>(t4 - t3);
//   std::cout << "fp32 : " << dt.count() << " microseconds " << std::endl;

//   C.print(std::cout);
//   C_fp32.print(std::cout);

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

TEST(nntrainer_Tensor, dot_gemm_768_768_768) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 768;

  int height_b = 768;
  int width_b = 768;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_raw(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_raw(batch, channel, height_b, width_b, t_type_nchw_fp16);
  nntrainer::Tensor C_raw(batch, channel, height, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  const float alpha = 1e-2;
  const int MOD = 10;

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
                        GEN_TEST_INPUT(A_raw, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B_raw, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);
  auto t1 = high_resolution_clock::now();
  nntrainer::Tensor C = A.dot(B, transA, transB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "function_to_assess : " << dt.count() << " microseconds "
            << std::endl;
  std::cout << "function_to_assess : "
            << (height * width * height_b) / dt.count() << " Gflops "
            << std::endl;

  t1 = high_resolution_clock::now();
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  t2 = high_resolution_clock::now();
  dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() << " microseconds " << std::endl;

  hgemm_raw_noTrans(height, width_b, width, A_raw.getData<__fp16>(), B_raw.getData<__fp16>(), C_raw.getData<__fp16>());

  C.print(std::cout);
  C_fp32.print(std::cout);
  C_raw.print(std::cout);
  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_raw.getData<__fp16>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_raw.getData<__fp16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_1440_1440_1440) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1440;
  int width = 1440;

  int height_b = 1440;
  int width_b = 1440;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_raw(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_raw(batch, channel, height_b, width_b, t_type_nchw_fp16);
  nntrainer::Tensor C_raw(batch, channel, height, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  const float alpha = 1e-2;
  const int MOD = 10;

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
                        GEN_TEST_INPUT(A_raw, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B_raw, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);
  auto t1 = high_resolution_clock::now();
  nntrainer::Tensor C = A.dot(B, transA, transB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "function_to_assess : " << dt.count() << " microseconds "
            << std::endl;
  std::cout << "function_to_assess : "
            << (height * width * height_b) / dt.count() << " Gflops "
            << std::endl;

  t1 = high_resolution_clock::now();
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  t2 = high_resolution_clock::now();
  dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() << " microseconds " << std::endl;

  hgemm_raw_noTrans(height, width_b, width, A_raw.getData<__fp16>(), B_raw.getData<__fp16>(), C_raw.getData<__fp16>());

  C.print(std::cout);
  C_fp32.print(std::cout);
  C_raw.print(std::cout);
  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_raw.getData<__fp16>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_raw.getData<__fp16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_2048_2048_2048) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 2048;
  int width = 2048;

  int height_b = 2048;
  int width_b = 2048;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_raw(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_raw(batch, channel, height_b, width_b, t_type_nchw_fp16);
  nntrainer::Tensor C_raw(batch, channel, height, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  const float alpha = 1e-2;
  const int MOD = 10;

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
                        GEN_TEST_INPUT(A_raw, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B_raw, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);
  auto t1 = high_resolution_clock::now();
  nntrainer::Tensor C = A.dot(B, transA, transB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "function_to_assess : " << dt.count() << " microseconds "
            << std::endl;
  std::cout << "function_to_assess : "
            << (height * width * height_b) / dt.count() << " Gflops "
            << std::endl;

  t1 = high_resolution_clock::now();
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  t2 = high_resolution_clock::now();
  dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() << " microseconds " << std::endl;

  hgemm_raw_noTrans(height, width_b, width, A_raw.getData<__fp16>(), B_raw.getData<__fp16>(), C_raw.getData<__fp16>());

  C.print(std::cout);
  C_fp32.print(std::cout);
  C_raw.print(std::cout);
  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_raw.getData<__fp16>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_raw.getData<__fp16>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_gemm_4096_4096_4096) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 4096;
  int width = 4096;

  int height_b = 4096;
  int width_b = 4096;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_raw(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B_raw(batch, channel, height_b, width_b, t_type_nchw_fp16);
  nntrainer::Tensor C_raw(batch, channel, height, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  const float alpha = 1e-2;
  const int MOD = 10;

  GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
                        GEN_TEST_INPUT(A_raw, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT_B(B_raw, ((i * (batch * height_b * channel) +
                        j * (batch * height_b) + k * (width_b) + l + 1) %
                       MOD) *
                        alpha);
  GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
                             j * (batch * height_b) + k * (width_b) + l + 1) %
                            MOD) *
                             alpha);
  auto t1 = high_resolution_clock::now();
  nntrainer::Tensor C = A.dot(B, transA, transB);
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "function_to_assess : " << dt.count() << " microseconds "
            << std::endl;
  std::cout << "function_to_assess : "
            << (height * width * height_b) / dt.count() << " Gflops "
            << std::endl;

  t1 = high_resolution_clock::now();
  nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  t2 = high_resolution_clock::now();
  dt = duration_cast<microseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() << " microseconds " << std::endl;

  // hgemm_raw_noTrans(height, width_b, width, A_raw.getData<__fp16>(), B_raw.getData<__fp16>(), C_raw.getData<__fp16>());

  C.print(std::cout);
  C_fp32.print(std::cout);
  // C_raw.print(std::cout);
  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

// TEST(nntrainer_Tensor, dot_gemm__uint16_1200_1200_1200) {
//   /// @note GEMM : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 54;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 96000;

//   bool transA = false;
//   bool transB = false;

//   int M = height;
//   int K = width;
//   int N = width_b;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

//   const float alpha = 1e-2;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1) %
//                             MOD) *
//                              alpha);
//   // uint16 tensor data generation
//   std::vector<uint16_t> u16_A;
//   std::vector<uint16_t> u16_B;
//   std::vector<uint16_t> u16_C;
//   std::vector<uint16_t> u16_C_raw;
//   // for (int i = 0; i < batch; ++i) {
//   //   for (int j = 0; j < channel; ++j) {
//   //     for (int k = 0; k < height; ++k) {
//   //       for (int l = 0; l < width; ++l) {
//   //         u16_A.push_back(((i+1)*(j+1)*(k+1)*(l+1))%MOD);
//   //         u16_B.push_back(((i+2)*(j+2)*(k+2)*(l+2))%MOD);
//   //         u16_C.push_back(0U);
//   //         u16_C_raw.push_back(0U);
//   //       }
//   //     }
//   //   }
//   // }
//   for (int i = 0; i < M; ++i) {
//     for (int j = 0; j < K; ++j) {
//       u16_A.push_back(((i + 1) * (j + 1)) % MOD);
//     }
//   }
//   for (int j = 0; j < K; ++j) {
//     for (int k = 0; k < N; ++k) {
//       u16_B.push_back(((j + 2) * (k + 2)) % MOD);
//     }
//   }
//   for (int i = 0; i < M; ++i) {
//     for (int k = 0; k < N; ++k) {
//       u16_C.push_back(0U);
//       u16_C_raw.push_back(0U);
//     }
//   }
//   auto tu1 = high_resolution_clock::now();
//   nntrainer::neon::uhgemm_noTrans(u16_A.data(), u16_B.data(), u16_C.data(),
//                                   M, N, K);
//   auto tu2 = high_resolution_clock::now();
//   auto dtu = duration_cast<microseconds>(tu2 - tu1);
//   std::cout << "uhgemm_noTrans : " << dtu.count() << " microseconds "
//             << std::endl;

//   auto t1 = high_resolution_clock::now();
//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   auto t2 = high_resolution_clock::now();
//   auto dt = duration_cast<microseconds>(t2 - t1);
//   std::cout << "hgemm_noTrans : " << dt.count() << " microseconds "
//             << std::endl;

//   t1 = high_resolution_clock::now();
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
//   t2 = high_resolution_clock::now();
//   dt = duration_cast<microseconds>(t2 - t1);
//   std::cout << "fp32 : " << dt.count() << " microseconds " << std::endl;

//   for (unsigned int m = 0; m < M; ++m) {
//     for (unsigned int n = 0; n < N; ++n) {
//       uint16_t c = 0;
//       for (unsigned int k = 0; k < K; ++k) {
//         uint16_t a, b;
//         a = u16_A[k + m * K];
//         b = u16_B[n + k * N];

//         // a = u16_A[m*K  + k];
//         // b = u16_B[k*N + n];

//         c += a * b;
//       }
//       u16_C_raw[m * N + n] += c;
//     }
//   }

//   C.print(std::cout);
//   C_fp32.print(std::cout);
//   for (int i = 0; i < u16_C.size() % MOD + 5; ++i) {
//     // std::cout<< "u16_A[i] : " << u16_A[i] << std::endl;
//     // std::cout << "u16_B[i] : "<< u16_B[i] << std::endl;
//     std::cout << "u16_C[i] : " << u16_C[i] << "\t";
//     std::cout << "u16_C_raw[i] : " << u16_C_raw[i] << std::endl;
//   }


//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemm__uint16_uint) {
//   /// @note GEMM : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 6;
//   int width = 4;

//   int height_b = 4;
//   int width_b = 16;

//   bool transA = false;
//   bool transB = false;

//   int M = height;
//   int K = width;
//   int N = width_b;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

//   const float alpha = 1e-2;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1) %
//                             MOD) *
//                              alpha);
//   // uint16 tensor data generation
//   std::vector<uint16_t> u16_A;
//   std::vector<uint16_t> u16_B;
//   std::vector<uint16_t> u16_C;
//   std::vector<uint16_t> u16_C_raw;
//   for (int i = 0; i < M; ++i) {
//     for (int j = 0; j < K; ++j) {
//       u16_A.push_back(1);
//     }
//   }
//   for (int j = 0; j < K; ++j) {
//     for (int k = 0; k < N; ++k) {
//       u16_B.push_back(2);
//     }
//   }
//   for (int i = 0; i < M; ++i) {
//     for (int k = 0; k < N; ++k) {
//       u16_C.push_back(0U);
//       u16_C_raw.push_back(0U);
//     }
//   }
//   auto tu1 = high_resolution_clock::now();
//   nntrainer::neon::uhgemm_noTrans(u16_A.data(), u16_B.data(), u16_C.data(),
//                                   M, N, K);
//   auto tu2 = high_resolution_clock::now();
//   auto dtu = duration_cast<microseconds>(tu2 - tu1);
//   std::cout << "uhgemm_noTrans : " << dtu.count() << " microseconds "
//             << std::endl;

//   auto t1 = high_resolution_clock::now();
//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   auto t2 = high_resolution_clock::now();
//   auto dt = duration_cast<microseconds>(t2 - t1);
//   std::cout << "hgemm_noTrans : " << dt.count() << " microseconds "
//             << std::endl;

//   t1 = high_resolution_clock::now();
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
//   t2 = high_resolution_clock::now();
//   dt = duration_cast<microseconds>(t2 - t1);
//   std::cout << "fp32 : " << dt.count() << " microseconds " << std::endl;

//   for (unsigned int m = 0; m < M; ++m) {
//     for (unsigned int n = 0; n < N; ++n) {
//       uint16_t c = 0;
//       for (unsigned int k = 0; k < K; ++k) {
//         uint16_t a, b;
//         a = u16_A[k + m * K];
//         b = u16_B[n + k * N];

//         // a = u16_A[m + k * K];
//         // b = u16_B[k + n * N];

//         c += a * b;
//       }
//       u16_C_raw[m * N + n] += c;
//     }
//   }

//   for (int i = 0; i < u16_C.size(); ++i) {
//     // std::cout<< "u16_A[i] : " << u16_A[i] << std::endl;
//     // std::cout << "u16_B[i] : "<< u16_B[i] << std::endl;
//     // std::cout << "u16_C[i] : " << u16_C[i] << "\t";
//     // std::cout << "u16_C_raw[i] : " << u16_C_raw[i] << std::endl;
//   }
//   EXPECT_EQ(u16_C, u16_C_raw);



//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemm_256_256_256) {
//   /// @note GEMM : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 256;
//   int width = 256;

//   int height_b = 256;
//   int width_b = 256;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
//   t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1)
//                              %
//                             MOD) *
//                              alpha);
//   auto t1 = high_resolution_clock::now();
//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   auto t2 = high_resolution_clock::now();
//   auto dt = duration_cast<microseconds>(t2 - t1);
//     std::cout << "fp16 : " <<  dt.count()
//           << " microseconds " << std::endl;
//   std::cout << "function_to_assess : " << (height * width * height_b) /
//   dt.count()
//           << " Tflops " << std::endl;

//   t1 = high_resolution_clock::now();
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
//   t2 = high_resolution_clock::now();
//   dt = duration_cast<microseconds>(t2 - t1);
//     std::cout << "fp32 : " <<  dt.count()
//           << " microseconds " << std::endl;

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemm_50_768_96000) {
//   /// @note GEMM : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 50;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 96000;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
//   t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1)
//                              %
//                             MOD) *
//                              alpha);
//     auto t1 = high_resolution_clock::now();
//   nntrainer::Tensor C = A.dot(B, transA, transB);
//     auto t2 = high_resolution_clock::now();
// auto dt = duration_cast<microseconds>(t2 - t1);
//     std::cout << "function_to_assess : " <<  dt.count()
//           << " microseconds " << std::endl;
// std::cout << "function_to_assess : " << (height * width * height_b) /
// dt.count()
//         << " Tflops " << std::endl;

//     t1 = high_resolution_clock::now();
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);
//     t2 = high_resolution_clock::now();
//   dt = duration_cast<microseconds>(t2 - t1);
//     std::cout << "fp32 : " <<  dt.count()
//           << " microseconds " << std::endl;

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemm_50_768_48000) {
//   /// @note GEMM : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 50;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 48000;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
//   t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1)
//                              %
//                             MOD) *
//                              alpha);

//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemm_50_768_20000) {
//   /// @note GEMM : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 50;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 20000;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
//   t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1)
//                              %
//                             MOD) *
//                              alpha);

//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemv_768_96000) {
//   /// @note GEMV : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 1;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 96000;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
//   t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1)
//                              %
//                             MOD) *
//                              alpha);

//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemv_768_48000) {
//   /// @note GEMV : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 1;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 48000;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
//   t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1)
//                              %
//                             MOD) *
//                              alpha);

//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_gemv_768_20000) {
//   /// @note GEMV : A X B = C
//   int batch = 1;
//   int channel = 1;
//   int height = 1;
//   int width = 768;

//   int height_b = 768;
//   int width_b = 20000;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
//   t_type_nchw_fp32);

//   const float alpha = 1e-1;
//   const int MOD = 10;

//   GEN_TEST_INPUT(A, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT_B(B, ((i * (batch * height_b * channel) +
//                         j * (batch * height_b) + k * (width_b) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(A_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);
//   GEN_TEST_INPUT_B(B_fp32, ((i * (batch * height_b * channel) +
//                              j * (batch * height_b) + k * (width_b) + l + 1)
//                              %
//                             MOD) *
//                              alpha);

//   nntrainer::Tensor C = A.dot(B, transA, transB);
//   nntrainer::Tensor C_fp32 = A_fp32.dot(B_fp32, transA, transB);

//   float mseErrorNeon =
//     mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

TEST(nntrainer_Tensor, inv_sqrt_i_p) {
  int batch = 4;
  int channel = 10;
  int height = 10;
  int width = 10;

  const int MOD = 10;
  const float eps = 1e-3;

  nntrainer::Tensor input(batch, channel, height, width,
                          nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16);
  GEN_TEST_INPUT(input, (i * (batch * height) + j * (width) + k) % MOD + 1);

  nntrainer::Tensor ground_truth(batch, channel, height, width,
                                 nntrainer::Tformat::NCHW,
                                 nntrainer::Tdatatype::FP32);
  GEN_TEST_INPUT(ground_truth,
                 (i * (batch * height) + j * (width) + k) % MOD + 1);

  input.inv_sqrt_i();

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          ground_truth.setValue(
            b, c, h, w, 1 / std::sqrt(ground_truth.getValue(b, c, h, w)));
        }
      }
    }
  }

  bool flag = true;

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          double err = std::abs(input.getValue<__fp16>(b, c, h, w) -
                                ground_truth.getValue(b, c, h, w));

          if (err > eps) {
            flag = false;
            std::cout << input.getValue<__fp16>(b, c, h, w) << " VS "
                      << ground_truth.getValue(b, c, h, w) << std::endl;
          }
        }
      }
    }
  }

  EXPECT_EQ(flag, true);
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
