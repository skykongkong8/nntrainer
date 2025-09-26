
/**
 * @file threading.cpp
 * @author Sungsik Kong (ss.kong@samsung.com)
 * @bug    No known bugs except for NYI items
 * @brief Copied file from MLAS (https://github.com/microsoft/MLAS)
 * @date 2025-09-29
 *
 */
/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    threading.cpp

Abstract:

    This module implements platform specific threading support.

--*/

#include <system_error>

#ifdef error_category
#undef error_category
#endif
#ifdef system_category
#undef system_category
#endif
#ifdef generic_category
#undef generic_category
#endif

#include "mlasi.h"

void MlasExecuteThreaded(MLAS_THREADED_ROUTINE *ThreadedRoutine, void *Context,
                         ptrdiff_t Iterations, MLAS_THREADPOOL *ThreadPool) {
  //
  // Execute the routine directly if only one iteration is specified.
  //

  if (Iterations == 1) {
    ThreadedRoutine(Context, 0);
    return;
  }

// #if defined(BUILD_MLAS_NO_ONNXRUNTIME)
#if 1
  MLAS_UNREFERENCED_PARAMETER(ThreadPool);

  //
  // Fallback to OpenMP or a serialized implementation.
  //

  //
  // Execute the routine for the specified number of iterations.
  //
  for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
    ThreadedRoutine(Context, tid);
  }
#else
  //
  // Schedule the threaded iterations using the thread pool object.
  //

  MLAS_THREADPOOL::TrySimpleParallelFor(
    ThreadPool, Iterations,
    [&](ptrdiff_t tid) { ThreadedRoutine(Context, tid); });
#endif
}

void MlasTrySimpleParallel(
  MLAS_THREADPOOL *ThreadPool, const std::ptrdiff_t Iterations,
  const std::function<void(std::ptrdiff_t tid)> &Work) {
  //
  // Execute the routine directly if only one iteration is specified.
  //
  if (Iterations == 1) {
    Work(0);
    return;
  }

// #if defined(BUILD_MLAS_NO_ONNXRUNTIME)
#if 1
  MLAS_UNREFERENCED_PARAMETER(ThreadPool);

  //
  // Fallback to OpenMP or a serialized implementation.
  //

  //
  // Execute the routine for the specified number of iterations.
  //
  for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
    Work(tid);
  }
#else
  //
  // Schedule the threaded iterations using the thread pool object.
  //

  MLAS_THREADPOOL::TrySimpleParallelFor(ThreadPool, Iterations, Work);
#endif
}

void MlasTryBatchParallel(MLAS_THREADPOOL *ThreadPool,
                          const std::ptrdiff_t Iterations,
                          const std::function<void(std::ptrdiff_t tid)> &Work) {
  //
  // Execute the routine directly if only one iteration is specified.
  //
  if (Iterations == 1) {
    Work(0);
    return;
  }

// #if defined(BUILD_MLAS_NO_ONNXRUNTIME)
#if 1
  MLAS_UNREFERENCED_PARAMETER(ThreadPool);

  //
  // Fallback to OpenMP or a serialized implementation.
  //

  //
  // Execute the routine for the specified number of iterations.
  //
  for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
    Work(tid);
  }
#else
  //
  // Schedule the threaded iterations using the thread pool object.
  //

  MLAS_THREADPOOL::TryBatchParallelFor(ThreadPool, Iterations, Work, 0);
#endif
}