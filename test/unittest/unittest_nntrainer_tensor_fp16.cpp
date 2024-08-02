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
using std::chrono::microseconds; // or microseconds
using std::chrono::milliseconds; // or microseconds
using std::chrono::nanoseconds;  // or microseconds
using std::chrono::seconds;      // or microseconds
#include <nntrainer_error.h>
#include <sys/time.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <time.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))


TEST(nntrainer_Tensor, dot_gemm_BM) {
  /// @note GEMM : A X B = C
  const unsigned int TC = 30;
  const unsigned int num_dot_run = 100;
  const unsigned int BM_dim_index = 96;
  const unsigned int dim_min = BM_dim_index * 1;
  const unsigned int dim_max = BM_dim_index * TC;

  int batch = 1;
  int channel = 1;
  int height = dim_min;
  int width = dim_min;

  int height_b = dim_min;
  int width_b = dim_min;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_fp16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  for (unsigned int tc = 0; tc < TC; ++tc) {
    double gflops = 2.0 * height * width * width_b * 1.0e-09;
    double gflops32 = 2.0 * height * width * width_b * 1.0e-09;

    nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_fp16);
    nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_fp16);

    nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
    nntrainer::Tensor B_fp32(batch, channel, height_b, width_b,
                             t_type_nchw_fp32);

    const float alpha = 1e-1;
    const int MOD = 10;

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
    nntrainer::Tensor C;
    auto t1 = high_resolution_clock::now();
    for (unsigned int ndr = 0; ndr < num_dot_run; ++ndr) {
      C = A.dot(B, transA, transB);
    }
    auto t2 = high_resolution_clock::now();
    auto dt = duration_cast<nanoseconds>(t2 - t1);
    double avg_latency = ((double)dt.count()) / num_dot_run;
    gflops = gflops / avg_latency;

    nntrainer::Tensor C_fp32;
    t1 = high_resolution_clock::now();
    for (unsigned int ndr = 0; ndr < num_dot_run; ++ndr) {
     C_fp32 = A_fp32.dot(B_fp32, transA, transB);
    }
    t2 = high_resolution_clock::now();
    dt = duration_cast<nanoseconds>(t2 - t1);
    double avg_latency32 = ((double)dt.count()) / num_dot_run;
    gflops32 = gflops32 / avg_latency32;

    float mseErrorNeon =
      mse<__fp16>(C.getData<__fp16>(), C_fp32.getData<float>(), C.size());

    double cosSimNeon = cosine_similarity<__fp16>(
      C.getData<__fp16>(), C_fp32.getData<float>(), C.size());
    std::cout << std::setprecision(8);
    std::cout<< std::fixed << "[INFO] | "
              << "Latency: " << avg_latency << " latency32: " << avg_latency32 << " Dim: " << height
              << std::endl;

    const float epsilon = 1e-3 * width;

    EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
    EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);

    height += BM_dim_index;
    width += BM_dim_index;

    height_b += BM_dim_index;
    width_b += BM_dim_index;
  }
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
