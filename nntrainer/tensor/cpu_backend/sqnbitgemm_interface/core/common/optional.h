/**
 * @file optional.h
 * @author Sungsik Kong (ss.kong@samsung.com)
 * @bug    No known bugs except for NYI items
 * @brief Copied file from MLAS (https://github.com/microsoft/MLAS)
 * @date 2025-09-29
 *
 */
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <optional>

namespace onnxruntime {

using std::optional;

#ifndef ORT_NO_EXCEPTIONS
using std::bad_optional_access;
#endif

using std::nullopt;
using std::nullopt_t;

using std::in_place;
using std::in_place_t;

using std::make_optional;

} // namespace onnxruntime
