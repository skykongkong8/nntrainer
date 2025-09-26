/**
 * @file nntr_sqnbitgemm_threadpool.cpp
 * @author Sungsik Kong (ss.kong@samsung.com)
 * @bug    No known bugs except for NYI items
 * @brief Copied file from MLAS (https://github.com/microsoft/MLAS)
 * @date 2025-09-29
 *
 */
#include "test_util.h"

#if 0
// #include "./core/platform/threadpool.h"

MLAS_THREADPOOL* GetMlasThreadPool(void) {
  static auto threadpool = std::make_unique<onnxruntime::concurrency::ThreadPool>(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 2, true);
  return threadpool.get();
}

#else

MLAS_THREADPOOL *GetMlasThreadPool(void) { return nullptr; }

#endif
