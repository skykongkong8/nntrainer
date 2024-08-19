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

// TEST(nntrainer_Tensor, uint16_dot_gemm_BM) {
//   /// @note GEMM : A X B = C
//   const unsigned int TC = 30;
//   const unsigned int num_dot_run = 100;
//   const unsigned int BM_dim_index = 96;
//   const unsigned int dim_min = BM_dim_index * 1;
//   const unsigned int dim_max = BM_dim_index * TC;

//   int batch = 1;
//   int channel = 1;
//   int height = dim_min;
//   int width = dim_min;

//   int height_b = dim_min;
//   int width_b = dim_min;

//   bool transA = false;
//   bool transB = false;

//   nntrainer::TensorDim::TensorType t_type_nchw_UINT16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::UINT16};

//   nntrainer::TensorDim::TensorType t_type_nchw_FP16 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP16};

//   nntrainer::TensorDim::TensorType t_type_nchw_fp32 = {
//     nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32};

//   nntrainer::Tensor A(batch, channel, height, width, t_type_nchw_UINT16);
//   nntrainer::Tensor B(batch, channel, height_b, width_b, t_type_nchw_UINT16);

//   nntrainer::Tensor A_fp32(batch, channel, height, width, t_type_nchw_fp32);
//   nntrainer::Tensor B_fp32(batch, channel, height_b, width_b, t_type_nchw_fp32);

//   nntrainer::Tensor A_fp16(batch, channel, height, width, t_type_nchw_FP16);
//   nntrainer::Tensor B_fp16(batch, channel, height_b, width_b, t_type_nchw_FP16);

//   for (unsigned int tc = 0; tc < TC; ++tc){

//   GEN_TEST_INPUT_RAND(A_fp32, 0, 5);
//   GEN_TEST_INPUT_RAND_B(B_fp32, 0, 5);

//   for (int b = 0; b < batch; b++) {
//     for (int c = 0; c < channel; c++) {
//       for (int h = 0; h < height; h++) {
//         for (int w = 0; w < width; w++) {
//           A_fp32.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
//         }
//       }
//     }
//   }

//   for (int b = 0; b < batch; b++) {
//     for (int c = 0; c < channel; c++) {
//       for (int h = 0; h < height_b; h++) {
//         for (int w = 0; w < width_b; w++) {
//           B_fp32.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
//         }
//       }
//     }
//   }

//   // A_fp32.print(std::cout);
//   // B_fp32.print(std::cout);

//   for (int b = 0; b < batch; b++) {
//     for (int c = 0; c < channel; c++) {
//       for (int h = 0; h < height; h++) {
//         for (int w = 0; w < width; w++) {
//           A.setValue(b, c, h, w, int(A_fp32.getValue<float>(b, c, h, w)));
//         }
//       }
//     }
//   }

//   for (int b = 0; b < batch; b++) {
//     for (int c = 0; c < channel; c++) {
//       for (int h = 0; h < height_b; h++) {
//         for (int w = 0; w < width_b; w++) {
//           B.setValue(b, c, h, w, int(B_fp32.getValue<float>(b, c, h, w)));
//         }
//       }
//     }
//   }

//   A.print(std::cout);
//   B.print(std::cout);

//   A_fp16.copyData(A_fp32);
//   B_fp16.copyData(B_fp32);

//   double gflops = 2.0 * height * width_b * width * 1.0e-09;

//   const int uTC = 1;

//   nntrainer::Tensor C;
//   nntrainer::Tensor C_fp32;
//   nntrainer::Tensor C_fp16;
//   auto t1 = high_resolution_clock::now();
//   for (int utc = 0; utc < uTC; ++utc) {
//     C = A.dot(B, transA, transB);
//   }
//   auto t2 = high_resolution_clock::now();
//   auto dt = duration_cast<nanoseconds>(t2 - t1);
//   std::cout << "uint16_t : " << dt.count() / TC << " ns "
//             << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

//   t1 = high_resolution_clock::now();
//   for (int utc = 0; utc < uTC; ++utc) {
//     C_fp32 = A_fp32.dot(B_fp32, transA, transB);
//   }
//   t2 = high_resolution_clock::now();
//   dt = duration_cast<nanoseconds>(t2 - t1);
//   std::cout << "fp32 : " << dt.count() / TC << " ns "
//             << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

//   t1 = high_resolution_clock::now();
//   for (int tc = 0; tc < TC; ++tc) {
//     C_fp16 = A_fp16.dot(B_fp16, transA, transB);
//   }
//   t2 = high_resolution_clock::now();
//   dt = duration_cast<nanoseconds>(t2 - t1);
//   std::cout << "fp16 : " << dt.count() / TC << " ns "
//             << ", gflops : " << 1e+9 * gflops / (dt.count() / TC) << std::endl;

//   float mseErrorNeon =
//     mse<uint16_t>(C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

//   double cosSimNeon = cosine_similarity<uint16_t>(
//     C.getData<uint16_t>(), C_fp32.getData<float>(), C.size());

//   float mcre = max_componentwise_relative_error<float, float, float, uint16_t>(
//     A_fp32.getData<float>(), B_fp32.getData<float>(), C_fp32.getData<float>(),
//     C.getData<uint16_t>(), A.size(), B.size(), C.size());

//   C.print(std::cout);
//   C_fp32.print(std::cout);

//   const float epsilon = 1e-3 * width;

//   EXPECT_IN_RANGE(mseErrorNeon, 0, epsilon);
//   EXPECT_IN_RANGE((float)cosSimNeon, 0.99, 1);
//   EXPECT_LE(mcre, 1e-5);

//     height += BM_dim_index;
//     width += BM_dim_index;

//     height_b += BM_dim_index;
//     width_b += BM_dim_index;
//   }
// }

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

  GEN_TEST_INPUT_RAND(A_fp32, 0, 9);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 9);

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

  GEN_TEST_INPUT_RAND(A_fp32, 0, 9);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 9);

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

  GEN_TEST_INPUT_RAND(A_fp32, 0, 9);
  GEN_TEST_INPUT_RAND_B(B_fp32, 0, 9);

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
