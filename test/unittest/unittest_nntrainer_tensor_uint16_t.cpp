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
#include <arm_neon.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

// TEST(nntrainer_Tensor, uint16_t_intrinsics_check) {
//   uint32x4_t a32 = {1, 25, 3, 4};
//   uint16x4_t b16 = {6, 4, 5, 3};

//   // uint16x4_t a16 = vreinterpret_u16_f16(vcvt_f16_f32(vcvtq_f32_u32(a32)));
//   // uint32x4_t b32 = vreinterpretq_u32_f32(vcvt_f32_f16(vcvt_f16_u16(b16)));

//   uint16x4_t a16 = vmovn_u32(a32);
//   uint32x4_t b32 = vmovl_u16(b16);

//   std::cout << "a : from u32 to u16\n";
//   for (int i = 0; i < 4; ++i) {
//     std::cout << a32[i] << "\t";
//   }
//   std::cout << std::endl;

//   for (int i = 0; i < 4; ++i) {
//     std::cout << a16[i] << "\t";
//   }
//   std::cout << std::endl;

//   std::cout << "b : from u16 to u32\n";
//   for (int i = 0; i < 4; ++i) {
//     std::cout << b16[i] << "\t";
//   }
//   std::cout << std::endl;

//   for (int i = 0; i < 4; ++i) {
//     std::cout << b32[i] << "\t";
//   }
//   std::cout << std::endl;
// }

static void ele_mul(unsigned int N, float* lhs, float* rhs, float* output){
    for (unsigned int n = 0; n < N; n +=4){
        float32x4_t a = vld1q_f32(&lhs[n]);
        float32x4_t b = vld1q_f32(&rhs[n]);
        vst1q_f32(&output[n], vmulq_f32(a,b));
        vst1q_f32(&output[n], vaddq_f32(vld1q_f32(&output[n]), vmulq_f32(a,b)));
        vst1q_f32(&output[n], vmulq_f32(vld1q_f32(&output[n]), vmulq_f32(a,b)));
        vst1q_f32(&output[n], vsubq_f32(vld1q_f32(&output[n]), vsubq_f32(a,b)));
    }
}

static void ele_mul(unsigned int N, uint16_t* lhs, uint16_t* rhs, uint16_t* output){
    for (unsigned int n = 0; n < N; n +=8){
        uint16x8_t a = vld1q_u16(&lhs[n]);
        uint16x8_t b = vld1q_u16(&rhs[n]);
        vst1q_u16(&output[n], vmulq_u16(a,b));
        vst1q_u16(&output[n], vaddq_u16(vld1q_u16(&output[n]), vmulq_u16(a,b)));
        vst1q_u16(&output[n], vmulq_u16(vld1q_u16(&output[n]), vmulq_u16(a,b)));
        vst1q_u16(&output[n], vsubq_u16(vld1q_u16(&output[n]), vsubq_u16(a,b)));
    }
}

static void ele_mul(unsigned int N, __fp16* lhs, __fp16* rhs, __fp16* output){
    for (unsigned int n = 0; n < N; n +=8){
        float16x8_t a = vld1q_f16(&lhs[n]);
        float16x8_t b = vld1q_f16(&rhs[n]);
        vst1q_f16(&output[n], vmulq_f16(a,b));
        vst1q_f16(&output[n], vaddq_f16(vld1q_f16(&output[n]), vmulq_f16(a,b)));
        vst1q_f16(&output[n], vmulq_f16(vld1q_f16(&output[n]), vmulq_f16(a,b)));
        vst1q_f16(&output[n], vsubq_f16(vld1q_f16(&output[n]), vsubq_f16(a,b)));
    }
}

TEST(nntrainer_Tensor, intrinsic_cmp) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1024;
  int width = 1024;

  unsigned int size = batch * channel * height * width;

  bool transA = false;
  bool transB = false;

  nntrainer::TensorDim::TensorType t_type_nchw_UINT16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16};

  nntrainer::TensorDim::TensorType t_type_nchw_FP16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_UINT16);
  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_FP16);

  GEN_TEST_INPUT_RAND(A_fp32, 0, 9);

  // A_fp32.print(std::cout);
  // B_fp32.print(std::cout);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  A.print(std::cout);
  A_fp16.copyData(A_fp32);

  const int TC = 1000;

  uint16_t* C = new uint16_t[size];
  float* C_fp32 = new float[size];
  __fp16* C_fp16 = new __fp16[size];
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    ele_mul(size, A.getData<uint16_t>(), A.getData<uint16_t>(), C);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "uint16_t : " << dt.count() << " ns " << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    ele_mul(size, A_fp32.getData<float>(), A_fp32.getData<float>(), C_fp32);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() << " ns " << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    ele_mul(size, A_fp16.getData<__fp16>(), A_fp16.getData<__fp16>(), C_fp16);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp16 : " << dt.count() << " ns " << std::endl;

  delete[] C;
  delete[] C_fp16;
  delete[] C_fp32;
}

TEST(nntrainer_Tensor, dot_gemm_srllm) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 86;
  int width = 768;

  int height_b = 768;
  int width_b = 2048;

  bool transA = false;
  bool transB = false;

  int min_val = 0;
  int max_val = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_UINT16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16};

  nntrainer::TensorDim::TensorType t_type_nchw_FP16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_UINT16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_UINT16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_FP16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_FP16);

  GEN_TEST_INPUT_RAND(A_fp32, min_val, max_val);
  GEN_TEST_INPUT_RAND_B(B_fp32, min_val, max_val);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A_fp32.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B_fp32.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  // A_fp32.print(std::cout);
  // B_fp32.print(std::cout);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  A.print(std::cout);
  B.print(std::cout);

  A_fp16.copyData(A_fp32);
  B_fp16.copyData(B_fp32);

  double gflops = 2.0 * height * width_b * width * 1.0e-09;

  const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
  nntrainer::Tensor C_fp16;
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C = A.dot(B, transA, transB);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "uint16_t : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp16 = A_fp16.dot(B_fp16, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp16 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  float mseErrorNeon =
    mse<uint16_t>(C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<uint16_t>(
    C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, uint16_t>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<uint16_t>(), A.size(), B.size(), C.size());

  C.print(std::cout);
  C_fp32.print(std::cout);

  auto eps = 1e-5;

  for (int b = 0; b < batch; b++) {
   for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width_b; w++) {
          double diff =std::abs(C.getValue<uint16_t>(b,c,h,w) - C_fp32.getValue<float>(b, c, h, w));
          if (diff > eps){
            std::cout << C.getValue<uint16_t>(b,c,h,w) << " VS " << C_fp32.getValue<float>(b, c, h, w) << std::endl;
          }
        }
      }
    }
  }

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
}

TEST(nntrainer_Tensor, dot_gemm_768) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 768;
  int width = 768;

  int height_b = 768;
  int width_b = 768;

  bool transA = false;
  bool transB = false;

  int min_val = 0;
  int max_val = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_UINT16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16};

  nntrainer::TensorDim::TensorType t_type_nchw_FP16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_UINT16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_UINT16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_FP16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_FP16);

  GEN_TEST_INPUT_RAND(A_fp32, min_val, max_val);
  GEN_TEST_INPUT_RAND_B(B_fp32, min_val, max_val);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A_fp32.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B_fp32.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  // A_fp32.print(std::cout);
  // B_fp32.print(std::cout);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  A.print(std::cout);
  B.print(std::cout);

  A_fp16.copyData(A_fp32);
  B_fp16.copyData(B_fp32);

  double gflops = 2.0 * height * width_b * width * 1.0e-09;

  const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
  nntrainer::Tensor C_fp16;
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C = A.dot(B, transA, transB);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "uint16_t : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp16 = A_fp16.dot(B_fp16, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp16 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  float mseErrorNeon =
    mse<uint16_t>(C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<uint16_t>(
    C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, uint16_t>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<uint16_t>(), A.size(), B.size(), C.size());

  C.print(std::cout);
  C_fp32.print(std::cout);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
  auto eps = 1e-5;

    for (int b = 0; b < batch; b++) {
   for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width_b; w++) {
          double diff =std::abs(C.getValue<uint16_t>(b,c,h,w) - C_fp32.getValue<float>(b, c, h, w));
          if (diff > eps){
            std::cout << C.getValue<uint16_t>(b,c,h,w) << " VS " << C_fp32.getValue<float>(b, c, h, w) << std::endl;
          }
        }
      }
    }
  }

  std::cout << "mseErrorNeon : " << mseErrorNeon << std::endl;
  std::cout << "cosSimNeon : " << cosSimNeon << std::endl;
  std::cout << "mcre : " << mcre << std::endl;
}

TEST(nntrainer_Tensor, dot_gemm_1024) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 1024;
  int width = 1024;

  int height_b = 1024;
  int width_b = 1024;

  bool transA = false;
  bool transB = false;

  int min_val = 0;
  int max_val = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_UINT16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16};

  nntrainer::TensorDim::TensorType t_type_nchw_FP16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_UINT16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_UINT16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_FP16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_FP16);

  GEN_TEST_INPUT_RAND(A_fp32, min_val, max_val);
  GEN_TEST_INPUT_RAND_B(B_fp32, min_val, max_val);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A_fp32.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B_fp32.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  // A_fp32.print(std::cout);
  // B_fp32.print(std::cout);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  A.print(std::cout);
  B.print(std::cout);

  A_fp16.copyData(A_fp32);
  B_fp16.copyData(B_fp32);

  double gflops = 2.0 * height * width_b * width * 1.0e-09;

  const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
  nntrainer::Tensor C_fp16;
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C = A.dot(B, transA, transB);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "uint16_t : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp16 = A_fp16.dot(B_fp16, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp16 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  float mseErrorNeon =
    mse<uint16_t>(C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<uint16_t>(
    C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, uint16_t>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<uint16_t>(), A.size(), B.size(), C.size());

  C.print(std::cout);
  C_fp32.print(std::cout);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
  auto eps = 1e-5;

  for (int b = 0; b < batch; b++) {
   for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width_b; w++) {
          double diff =std::abs(C.getValue<uint16_t>(b,c,h,w) - C_fp32.getValue<float>(b, c, h, w));
          if (diff > eps){
            std::cout << C.getValue<uint16_t>(b,c,h,w) << " VS " << C_fp32.getValue<float>(b, c, h, w) << std::endl;
          }
        }
      }
    }
  }

  std::cout << "mseErrorNeon : " << mseErrorNeon << std::endl;
  std::cout << "cosSimNeon : " << cosSimNeon << std::endl;
  std::cout << "mcre : " << mcre << std::endl;
}

TEST(nntrainer_Tensor, dot_gemm_96000) {
  /// @note GEMM : A X B = C
  int batch = 1;
  int channel = 1;
  int height = 96;
  int width = 768;

  int height_b = 768;
  int width_b = 96000;

  bool transA = false;
  bool transB = false;

  int min_val = 0;
  int max_val = 10;

  nntrainer::TensorDim::TensorType t_type_nchw_UINT16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16};

  nntrainer::TensorDim::TensorType t_type_nchw_FP16 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

  nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
    nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

  nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_UINT16);
  nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_UINT16);

  nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
  nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

  nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_FP16);
  nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_FP16);

  GEN_TEST_INPUT_RAND(A_fp32, min_val, max_val);
  GEN_TEST_INPUT_RAND_B(B_fp32, min_val, max_val);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A_fp32.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B_fp32.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  // A_fp32.print(std::cout);
  // B_fp32.print(std::cout);

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          A.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  for (int b = 0; b < batch; b++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height_b; h++) {
        for (int w = 0; w < width_b; w++) {
          B.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
        }
      }
    }
  }

  A.print(std::cout);
  B.print(std::cout);

  A_fp16.copyData(A_fp32);
  B_fp16.copyData(B_fp32);

  double gflops = 2.0 * height * width_b * width * 1.0e-09;

  const int TC = 100;

  nntrainer::Tensor C;
  nntrainer::Tensor C_fp32;
  nntrainer::Tensor C_fp16;
  auto t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C = A.dot(B, transA, transB);
  }
  auto t2 = high_resolution_clock::now();
  auto dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "uint16_t : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp32 = A_fp32.dot(B_fp32, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp32 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  t1 = high_resolution_clock::now();
  for (int tc = 0; tc < TC; ++tc) {
    C_fp16 = A_fp16.dot(B_fp16, transA, transB);
  }
  t2 = high_resolution_clock::now();
  dt = duration_cast<nanoseconds>(t2 - t1);
  std::cout << "fp16 : " << dt.count() / TC << " ns "
            << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

  float mseErrorNeon =
    mse<uint16_t>(C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  double cosSimNeon = cosine_similarity<uint16_t>(
    C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

  float mcre = max_componentwise_relative_error<float, float, float, uint16_t>(
    A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
    C.getData<uint16_t>(), A.size(), B.size(), C.size());

  C.print(std::cout);
  C_fp32.print(std::cout);

  const float epsilon = 1e-3 * width;

  EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
  EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
  EXPECT_LE(mcre, 1e-5);
  auto eps = 1e-5;

  for (int b = 0; b < batch; b++) {
   for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width_b; w++) {
          double diff =std::abs(C.getValue<uint16_t>(b,c,h,w) - C_fp32.getValue<float>(b, c, h, w));
          if (diff > eps){
            std::cout << C.getValue<uint16_t>(b,c,h,w) << " VS " << C_fp32.getValue<float>(b, c, h, w) << std::endl;
          }
        }
      }
    }
  }
  std::cout << "mseErrorNeon : " << mseErrorNeon << std::endl;
  std::cout << "cosSimNeon : " << cosSimNeon << std::endl;
  std::cout << "mcre : " << mcre << std::endl;
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
