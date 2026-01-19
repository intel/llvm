/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
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
