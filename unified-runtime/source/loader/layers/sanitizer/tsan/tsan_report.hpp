/*
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM
 * Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_report.hpp
 *
 */

#pragma once

#include "sanitizer_common/sanitizer_common.hpp"
#include "tsan_libdevice.hpp"

namespace ur_sanitizer_layer {
namespace tsan {

void ReportDataRace(const TsanErrorReport &Report, ur_kernel_handle_t Kernel);

} // namespace tsan
} // namespace ur_sanitizer_layer
