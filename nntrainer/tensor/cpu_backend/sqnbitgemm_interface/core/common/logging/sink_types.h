/**
 * @file sink_types.h
 * @author Sungsik Kong (ss.kong@samsung.com)
 * @bug    No known bugs except for NYI items
 * @brief Copied file from MLAS (https://github.com/microsoft/MLAS)
 * @date 2025-09-29
 *
 */
#pragma once

namespace onnxruntime {
namespace logging {
enum class SinkType { BaseSink, CompositeSink, EtwSink };
} // namespace logging
} // namespace onnxruntime
