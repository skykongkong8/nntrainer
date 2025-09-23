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
