/**
 * @file ort_mutex.h
 * @author Sungsik Kong (ss.kong@samsung.com)
 * @bug    No known bugs except for NYI items
 * @brief Copied file from MLAS (https://github.com/microsoft/MLAS)
 * @date 2025-09-29
 *
 */
#pragma once

#include <condition_variable>
#include <mutex>

namespace onnxruntime {
using OrtMutex = std::mutex;
using OrtCondVar = std::condition_variable;
} // namespace onnxruntime