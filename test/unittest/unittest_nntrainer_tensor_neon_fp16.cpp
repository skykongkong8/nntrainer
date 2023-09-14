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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

#define GEN_TEST_INPUT_T(input, equation_i_j_k_l) \
  do {                                            \
    for (int i = 0; i < batch; ++i) {             \
      for (int j = 0; j < channel; ++j) {         \
        for (int k = 0; k < height_t; ++k) {      \
          for (int l = 0; l < width_t; ++l) {     \
            float val = equation_i_j_k_l;         \
            input.setValue(i, j, k, l, val);      \
          }                                       \
        }                                         \
      }                                           \
    }                                             \
  } while (0)

// TEST(nntrainer_Tensor, add_i) {
//   int batch = 1;
//   int channel = 1;
//   int height = 2;
//   int width = 11;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor input_copy(batch, channel, height, width,
//   t_type_nchw_fp16); nntrainer::Tensor input_fp32(batch, channel, height,
//   width, t_type_nchw_fp32);

//   const float alpha = 1e-5;
//   const float epsilon = 1e-4;

//   GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
//                           j * (batch * height) * alpha + k * (width)*alpha +
//                           l + 1);
//   GEN_TEST_INPUT(input_copy, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);
//   GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);

//   // NEON fp16
//   int result = input.add_i(input_copy);

//   // fp32
//   result = input_fp32.add_i(input_fp32);

//   float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
//                                    input_fp32.getData<float>(),
//                                    input.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot) {

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   // conditions for fp16 sdot call:
//   // this->(batch * channel * height) = arg->(width) = 1;

//   size_t width = 23;

//   __fp16 a_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
//                      12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//   nntrainer::Tensor input(
//     nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);
//   __fp16 b_data[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
//                      12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//   nntrainer::Tensor input_2(
//     nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp16), b_data);

//   float a_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
//                          12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//   nntrainer::Tensor input_fp32(
//     nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);
//   float b_data_fp32[] = {0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11,
//                          12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//   nntrainer::Tensor input_fp32_2(
//     nntrainer::TensorDim(1, 1, width, 1, t_type_nchw_fp32), b_data_fp32);

//   nntrainer::Tensor result_neon;
//   nntrainer::Tensor result_fp32;

//   // NEON fp16
//   result_neon = input.dot(input_2, false, false);

//   // fp32
//   result_fp32 = input_fp32.dot(input_fp32_2, false, false);

//   float mseErrorNeon =
//     mse<__fp16>(result_neon.getData<__fp16>(), result_fp32.getData<float>(),
//                 result_neon.size());

//   double cosSimNeon =
//     cosine_similarity<__fp16>(result_neon.getData<__fp16>(),
//                               result_fp32.getData<float>(),
//                               result_neon.size());

//   const float epsilon = 1e-4;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, l2norm) {

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   size_t width = 23;

//   __fp16 a_data[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6, 0, 2.7,
//                      2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
//   nntrainer::Tensor input(
//     nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

//   float a_data_fp32[] = {0,   1.2, 2, 3.4, 4.1, 5.3, 2.9, 2.1, 1.4, 1.6,
//   0, 2.7,
//                          2.3, 1,   2, 1.1, 3.1, 1.1, 2.8, 3.2, 2,   3.6, 1};
//   nntrainer::Tensor input_fp32(
//     nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

//   __fp16 result_neon;
//   float result_fp32;

//   // NEON fp16
//   result_neon = input.l2norm();

//   // fp32
//   result_fp32 = input_fp32.l2norm();

//   // absolute error
//   const float epsilon = 1e-2;

//   EXPECT_NEAR(result_neon, result_fp32, epsilon);
// }

// TEST(nntrainer_Tensor, multiply_i) {
//   int batch = 1;
//   int channel = 1;
//   int height = 2;
//   int width = 11;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor input_copy(batch, channel, height, width,
//   t_type_nchw_fp16); nntrainer::Tensor input_fp32(batch, channel, height,
//   width, t_type_nchw_fp32);

//   const float alpha = 1e-5;
//   const float epsilon = 1e-4;

//   GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
//                           j * (batch * height) * alpha + k * (width)*alpha +
//                           l + 1);
//   GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);

//   // NEON fp16
//   int result = input.multiply_i(0.1);

//   // fp32
//   result = input_fp32.multiply_i(0.1);

//   float mseErrorNeon = mse<__fp16>(input.getData<__fp16>(),
//                                    input_fp32.getData<float>(),
//                                    input.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     input.getData<__fp16>(), input_fp32.getData<float>(), input.size());

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, copy) {
//   int batch = 1;
//   int channel = 1;
//   int height = 2;
//   int width = 11;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor input_fp32(batch, channel, height, width,
//   t_type_nchw_fp32);

//   const float alpha = 1e-5;
//   const float epsilon = 1e-4;

//   GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
//                           j * (batch * height) * alpha + k * (width)*alpha +
//                           l + 1);
//   GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);

//   nntrainer::Tensor output;
//   nntrainer::Tensor output_fp32;

//   // NEON fp16
//   output.copy(input);

//   // fp32
//   output_fp32.copy(input_fp32);

//   float mseErrorNeon = mse<__fp16>(output.getData<__fp16>(),
//                                    output_fp32.getData<float>(),
//                                    output.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     output.getData<__fp16>(), output_fp32.getData<float>(), output.size());

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE(cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, max_abs) {

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   size_t width = 25;

//   __fp16 a_data[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
//                      1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
//                      2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
//   nntrainer::Tensor input(
//     nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp16), a_data);

//   float a_data_fp32[] = {0,   1.2, 2,   3.4, 4.1, 5.3, 2.9, 2.1, 1.4,
//                          1.6, 0,   2.7, 2.3, 1,   2,   1.1, 3.1, 1.1,
//                          2.8, 3.2, 2,   3.6, 1,   2.8, 7.9};
//   nntrainer::Tensor input_fp32(
//     nntrainer::TensorDim(1, 1, 1, width, t_type_nchw_fp32), a_data_fp32);

//   __fp16 result_neon;
//   float result_fp32;

//   // NEON fp16
//   result_neon = input.max_abs();

//   // fp32
//   result_fp32 = input_fp32.max_abs();

//   // absolute error
//   const float epsilon = 1e-2;

//   EXPECT_NEAR(result_neon, result_fp32, epsilon);
// }

// TEST(nntrainer_Tensor, sum_sgemv_transpose) {
//   int batch = 3;
//   int channel = 2;
//   int height = 2;
//   int width = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor input_copy(batch, channel, height, width,
//   t_type_nchw_fp16); nntrainer::Tensor input_fp32(batch, channel, height,
//   width, t_type_nchw_fp32);

//   const float alpha = 1e-5;

//   GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
//                           j * (batch * height) * alpha + k * (width)*alpha +
//                           l + 1);
//   GEN_TEST_INPUT(input_copy, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);
//   GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);

//   nntrainer::Tensor result0 = input.sum(0);
//   nntrainer::Tensor result0_fp32 = input_fp32.sum(0);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(),
//     result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(),
//     result0.size());

//   const float epsilon = 1e-4;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, sum_sgemv) {
//   int batch = 3;
//   int channel = 2;
//   int height = 2;
//   int width = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor input_copy(batch, channel, height, width,
//   t_type_nchw_fp16); nntrainer::Tensor input_fp32(batch, channel, height,
//   width, t_type_nchw_fp32);

//   const float alpha = 1e-5;

//   GEN_TEST_INPUT(input, i * (batch * height * channel) * alpha +
//                           j * (batch * height) * alpha + k * (width)*alpha +
//                           l + 1);
//   GEN_TEST_INPUT(input_copy, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);
//   GEN_TEST_INPUT(input_fp32, i * (batch * height * channel) * alpha +
//                                j * (batch * height) * alpha +
//                                k * (width)*alpha + l + 1);

//   nntrainer::Tensor result0 = input.sum_by_batch();
//   nntrainer::Tensor result0_fp32 = input_fp32.sum_by_batch();

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(),
//     result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(),
//     result0.size());

//   const float epsilon = 1e-4;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transA) {
//   int batch = 1;
//   int channel = 1;
//   int height = 1440;
//   int width = 1024;

//   int height_t = 1440;
//   int width_t = 1440;

//   bool transA = true;
//   bool transB = false;
//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                         k * (width) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                              j * (batch * height) + k * (width) + l + 1) %
//                             MOD) *
//                              alpha);
//   // input.print(std::cout);
//   // input_fp32.print(std::cout);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   result0.print(std::cout);
//   result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * result0.size());
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transA_501) {
//   int batch = 1;
//   int channel = 1;
//   int height = 501;
//   int width = 501;

//   int height_t = 501;
//   int width_t = 80;

//   bool transA = true;
//   bool transB = false;
//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                         k * (width) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                              j * (batch * height) + k * (width) + l + 1) %
//                             MOD) *
//                              alpha);
//   // input.print(std::cout);
//   // input_fp32.print(std::cout);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   result0.print(std::cout);
//   result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * result0.size());
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transA_80) {
//   int batch = 1;
//   int channel = 1;
//   int height = 80;
//   int width = 501;

//   int height_t = 80;
//   int width_t = 501;

//   bool transA = true;
//   bool transB = false;
//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                         k * (width) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                              j * (batch * height) + k * (width) + l + 1) %
//                             MOD) *
//                              alpha);
//   // input.print(std::cout);
//   // input_fp32.print(std::cout);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   result0.print(std::cout);
//   result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * result0.size());
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transB) {
//   int batch = 1;
//   int channel = 1;
//   int height = 1024;
//   int width = 1440;

//   int height_t = 1440;
//   int width_t = 1440;

//   bool transA = false;
//   bool transB = true;
//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                         k * (width) + l + 1) %
//                        MOD) *
//                         alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                              j * (batch * height) + k * (width) + l + 1) %
//                             MOD) *
//                              alpha);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   result0.print(std::cout);
//   result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * height);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transB_501) {
//   int batch = 1;
//   int channel = 1;
//   int height = 501;
//   int width = 501;

//   int height_t = 80;
//   int width_t = 501;

//   bool transA = false;
//   bool transB = true;
//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input,
//                  ((i * (batch * height * channel) % MOD +
//                    j * (batch * height) % MOD + k * (width) % MOD + l + 1) %
//                   MOD) *
//                    alpha);

//   GEN_TEST_INPUT_T(
//     m, ((i * (batch * height_t * channel) % MOD + j * (batch * height_t) % MOD +
//          k * (width_t) % MOD + l + 1) %
//         MOD) *
//          alpha);

//   GEN_TEST_INPUT(input_fp32,
//                  ((i * (batch * height * channel) % MOD +
//                    j * (batch * height) % MOD + k * (width) % MOD + l + 1) %
//                   MOD) *
//                    alpha);
//   GEN_TEST_INPUT_T(
//     m_fp32, ((i * (batch * height_t * channel) % MOD +
//               j * (batch * height_t) % MOD + k * (width_t) % MOD + l + 1) %
//              MOD) *
//               alpha);

//   input_fp32.print(std::cout);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   result0.print(std::cout);
//   result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * height);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transB_80) {
//   int batch = 1;
//   int channel = 1;
//   int height = 501;
//   int width = 80;

//   int height_t = 501;
//   int width_t = 80;

//   bool transA = false;
//   bool transB = true;
//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   result0.print(std::cout);
//   result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * height);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }


TEST(nntrainer_Tensor, dot_sgemm_transB_80_naive) {
  int batch = 1;
  int channel = 1;
  int height = 501;
  int width = 80;

  int height_t = 501;
  int width_t = 80;

  bool transA = false;
  bool transB = true;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_naive(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m_naive(batch, channel, height_t, width_t, t_type_nchw_fp16);

  float alpha = -0.1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);
  GEN_TEST_INPUT(input_naive, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_naive, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_naive = input_naive.naive_dot(m_naive, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  const float eps = 1e-10;

  for (int i = 0; i < result0.batch(); ++i){
    for (int j = 0; j < result0.channel(); ++j){
      for(int k = 0; k < result0.height(); ++k){
        for(int l = 0; l < result0.width(); ++l){
          auto v1 = (result0.getValue<__fp16>(i,j,k,l));
          auto v2 = result0_naive.getValue<__fp16>(i,j,k,l);

          if (v1 != v2){
            std::cout << "v1 : " << float(v1) << " VS " << " v2 : " << float(v2) << std::endl;
          }
        }
      }
    }
  }

  std::cout.precision(10);
  // EXPECT_EQ(result0, result0_naive);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;



  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

// TEST(nntrainer_Tensor, dot_sgemm_transAB) {
//   int batch = 1;
//   int channel = 1;
//   int height = 1440;
//   int width = 1024;

//   int height_t = 1440;
//   int width_t = 1440;

//   bool transA = true;
//   bool transB = true;

//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);

//   // input.print(std::cout);
//   // input_fp32.print(std::cout);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   // result0.print(std::cout);
//   // result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * result0.size());
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transAB_501) {
//   int batch = 1;
//   int channel = 1;
//   int height = 501;
//   int width = 501;

//   int height_t = 80;
//   int width_t = 501;

//   bool transA = true;
//   bool transB = true;

//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width,
//   t_type_nchw_fp32); nntrainer::Tensor m_fp32(batch, channel, height_t,
//   width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);

//   // input.print(std::cout);
//   // input_fp32.print(std::cout);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(),
//     result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(),
//     result0.size());

//   // result0.print(std::cout);
//   // result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * result0.size());
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

// TEST(nntrainer_Tensor, dot_sgemm_transAB_80) {
//   int batch = 1;
//   int channel = 1;
//   int height = 80;
//   int width = 501;

//   int height_t = 501;
//   int width_t = 80;

//   bool transA = true;
//   bool transB = true;

//   int MOD = 10;

//   nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
//   nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

//   nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

//   float alpha = 1e-1;

//   GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
//                           j * (batch * height) + k * (width) + l + 1) %
//                          MOD) *
//                           alpha);
//   GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
//                       k * (width) + l + 1) %
//                      MOD) *
//                       alpha);
//   GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
//                                j * (batch * height) + k * (width) + l + 1) %
//                               MOD) *
//                                alpha);
//   GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
//                            j * (batch * height) + k * (width) + l + 1) %
//                           MOD) *
//                            alpha);

//   // input.print(std::cout);
//   // input_fp32.print(std::cout);

//   nntrainer::Tensor result0 = input.dot(m, transA, transB);
//   nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

//   float mseErrorNeon = mse<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   double cosSimNeon = cosine_similarity<__fp16>(
//     result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

//   // result0.print(std::cout);
//   // result0_fp32.print(std::cout);

//   const float epsilon = 1e-3;
//   std::cout << "mse : " << mseErrorNeon << std::endl;
//   std::cout << "cosine similarity : " << cosSimNeon << std::endl;
//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * result0.size());
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
// }

TEST(nntrainer_Tensor, dot_sgemm_no_trans) {
  int batch = 1;
  int channel = 1;
  int height = 520;
  int width = 2304;

  int height_t = 2304;
  int width_t = 2304;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

  float alpha = 1e-1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_sgemm_no_trans2) {
  int batch = 1;
  int channel = 1;
  int height = 1024;
  int width = 1440;

  int height_t = 1440;
  int width_t = 1440;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

  float alpha = 1e-1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * height_t);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_sgemm_no_501) {
  int batch = 1;
  int channel = 1;
  int height = 80;
  int width = 501;

  int height_t = 501;
  int width_t = 501;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

  float alpha = 1e-1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);
  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel) + j * (batch * height) +
                      k * (width) + l + 1) %
                     MOD) *
                      alpha);
  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_sgemm_no_80) {
  int batch = 1;
  int channel = 1;
  int height = 501;
  int width = 80;

  int height_t = 80;
  int width_t = 501;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

  float alpha = 1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_sgemm_no_80_924) {
  int batch = 1;
  int channel = 1;
  int height = 924;
  int width = 80;

  int height_t = 80;
  int width_t = 924;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

  float alpha = 1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}


TEST(nntrainer_Tensor, dot_sgemm_no_924) {
  int batch = 1;
  int channel = 1;
  int height = 924;
  int width = 924;

  int height_t = 924;
  int width_t = 80;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor m_fp32(batch, channel, height_t, width_t, t_type_nchw_fp32);

  float alpha = 1e-1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT(input_fp32, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_fp32, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_fp32 = input_fp32.dot(m_fp32, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_fp32.getData<float>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}
#include <algorithm>
TEST(nntrainer_Tensor, dot_sgemm_no_924_naive) {
  int batch = 1;
  int channel = 1;
  int height = 924;
  int width = 924;

  int height_t = 924;
  int width_t = 80;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_naive(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m_naive(batch, channel, height_t, width_t, t_type_nchw_fp16);

  float alpha = -0.1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);
  GEN_TEST_INPUT(input_naive, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_naive, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_naive = input_naive.naive_dot(m_naive, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  const float eps = 1e-10;

  for (int i = 0; i < result0.batch(); ++i){
    for (int j = 0; j < result0.channel(); ++j){
      for(int k = 0; k < result0.height(); ++k){
        for(int l = 0; l < result0.width(); ++l){
          auto v1 = (result0.getValue<__fp16>(i,j,k,l));
          auto v2 = result0_naive.getValue<__fp16>(i,j,k,l);

          // if (std::abs((float)v1-(float)v2) > eps){
          if ((v1 != v2)){
            std::cout << "v1 : " << float(v1) << " VS " << " v2 : " << float(v2) << std::endl;
          }
        }
      }
    }
  }

  std::cout.precision(10);
  EXPECT_EQ(result0, result0_naive);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;



  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}

TEST(nntrainer_Tensor, dot_sgemm_no_80_naive) {
  int batch = 1;
  int channel = 1;
  int height = 924;
  int width = 80;

  int height_t = 80;
  int width_t = 924;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_naive(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m_naive(batch, channel, height_t, width_t, t_type_nchw_fp16);

  float alpha = -0.1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);
  GEN_TEST_INPUT(input_naive, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_naive, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_naive = input_naive.naive_dot(m_naive, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  const float eps = 1e-10;

  for (int i = 0; i < result0.batch(); ++i){
    for (int j = 0; j < result0.channel(); ++j){
      for(int k = 0; k < result0.height(); ++k){
        for(int l = 0; l < result0.width(); ++l){
          auto v1 = (result0.getValue<__fp16>(i,j,k,l));
          auto v2 = result0_naive.getValue<__fp16>(i,j,k,l);

          if (v1 != v2){
            std::cout << "v1 : " << float(v1) << " VS " << " v2 : " << float(v2) << std::endl;
          }
        }
      }
    }
  }

  std::cout.precision(10);
  EXPECT_EQ(result0, result0_naive);

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  const float epsilon = 1e-2;

  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;



  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
}


TEST(nntrainer_Tensor, dot_sgemm_no_501_naive) {
  int batch = 1;
  int channel = 1;
  int height = 501;
  int width = 80;

  int height_t = 80;
  int width_t = 501;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_naive(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m_naive(batch, channel, height_t, width_t, t_type_nchw_fp16);

  float alpha = -0.1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);
  GEN_TEST_INPUT(input_naive, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_naive, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_naive = input_naive.naive_dot(m_naive, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  const float eps = 1e-10;

  for (int i = 0; i < result0.batch(); ++i){
    for (int j = 0; j < result0.channel(); ++j){
      for(int k = 0; k < result0.height(); ++k){
        for(int l = 0; l < result0.width(); ++l){
          auto v1 = (result0.getValue<__fp16>(i,j,k,l));
          auto v2 = result0_naive.getValue<__fp16>(i,j,k,l);

          if (v1 != v2){
            std::cout << "v1 : " << float(v1) << " VS " << " v2 : " << float(v2) << std::endl;
          }
        }
      }
    }
  }

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  const float epsilon = 1e-2;

  std::cout.precision(10);
  EXPECT_EQ(result0, result0_naive);


  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;



  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);

}

TEST(nntrainer_Tensor, dot_sgemm_no_2304_naive) {
  int batch = 1;
  int channel = 1;
  int height = 520;
  int width = 2304;

  int height_t = 2304;
  int width_t = 2304;

  bool transA = false;
  bool transB = false;

  int MOD = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor input(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m(batch, channel, height_t, width_t, t_type_nchw_fp16);

  nntrainer::Tensor input_naive(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor m_naive(batch, channel, height_t, width_t, t_type_nchw_fp16);

  float alpha = -0.1;

  GEN_TEST_INPUT(input, ((i * (batch * height * channel) +
                          j * (batch * height) + k * (width) + l + 1) %
                         MOD) *
                          alpha);

  GEN_TEST_INPUT_T(m, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);
  GEN_TEST_INPUT(input_naive, ((i * (batch * height * channel) +
                               j * (batch * height) + k * (width) + l + 1) %
                              MOD) *
                               alpha);
  GEN_TEST_INPUT_T(m_naive, ((i * (batch * height * channel * 10) +
                           j * (batch * height) + k * (width) + l + 1) %
                          MOD) *
                           -alpha);

  // input.print(std::cout);
  // input_fp32.print(std::cout);

  nntrainer::Tensor result0 = input.dot(m, transA, transB);
  nntrainer::Tensor result0_naive = input_naive.naive_dot(m_naive, transA, transB);

  // result0.print(std::cout);
  // result0_fp32.print(std::cout);

  const float eps = 1e-10;

  for (int i = 0; i < result0.batch(); ++i){
    for (int j = 0; j < result0.channel(); ++j){
      for(int k = 0; k < result0.height(); ++k){
        for(int l = 0; l < result0.width(); ++l){
          auto v1 = (result0.getValue<__fp16>(i,j,k,l));
          auto v2 = result0_naive.getValue<__fp16>(i,j,k,l);

          if (v1 != v2){
            std::cout << "v1 : " << float(v1) << " VS " << " v2 : " << float(v2) << std::endl;
          }
        }
      }
    }
  }

  float mseErrorNeon = mse<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    result0.getData<__fp16>(), result0_naive.getData<__fp16>(), result0.size());

  const float epsilon = 1e-2;

  std::cout.precision(10);
  EXPECT_EQ(result0, result0_naive);


  std::cout << "mse : " << mseErrorNeon << std::endl;
  std::cout << "cosine similarity : " << cosSimNeon << std::endl;



  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon * width);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);

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