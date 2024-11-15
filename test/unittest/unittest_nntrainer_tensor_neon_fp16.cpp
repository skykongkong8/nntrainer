// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file        unittest_nntrainer_tensor_neon_fp16.cpp
 * @date        03 August 2023
 * @brief       Unit test utility for tensor with NEON __fp16 support for ARM.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Debadri Samaddar <s.debadri@samsung.com>
 * @author      Sungsik Kong <ss.kong@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include "util_func.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
using std::chrono::nanoseconds; // or microseconds
using std::chrono::microseconds; // or microseconds
using std::chrono::milliseconds; // or microseconds
using std::chrono::seconds; // or microseconds
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))


TEST(nntrainer_Tensor, dot_gemm_1) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_8) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 8;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_16) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 16;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_32) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 32;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_64) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 64;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_128) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 128;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_256) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 256;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_512) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 512;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_768) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_1024) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1024;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_1440) {
  /// @note GEMV : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1440;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 1);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 1);

  A.copyData(A_fp32);
  B.copyData(B_fp32);
double  gflops = 2.0 * height * width_b * width * 1.0e-09;

const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
auto t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C = A.dot(B, transA, transB);
}
auto t2 = high_resolution_clock::now();
auto dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp16 : " << dt.count() / TC
        << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;

t1 = high_resolution_clock::now();
for (int tc = 0; tc < TC; ++tc){
  C_fp32 = A_fp32.dot(B_fp32, transA, transB);
}
t2 = high_resolution_clock::now();
dt = duration_cast<nanoseconds>(t2 - t1);
std::cout << "fp32 : " << dt.count() / TC
                << " ns " << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) <<std::endl;


  float mseErrorNeon =
    mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<__fp16>(
    C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, __fp16>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<__fp16>(), A.size(), B.size(), C.size());

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

GTEST_API_ int main(int argc, char **argv) {
  int result = -1;

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

  return result;
}
