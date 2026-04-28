/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_report.hpp
 *
 */

#pragma once

#include "unified-runtime/ur_api.h"

namespace ur_sanitizer_layer {

struct MsanErrorReport;

namespace msan {

// Abort the program if the return value is true
bool ReportUsesUninitializedValue(const MsanErrorReport &Report,
                                  ur_kernel_handle_t Kernel);

} // namespace msan
} // namespace ur_sanitizer_layer
