/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file tsan_report.cpp
 *
 */

#include "tsan_report.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace tsan {

void ReportDataRace(const TsanErrorReport &Report, ur_kernel_handle_t Kernel) {
  const char *File = Report.File[0] ? Report.File : "<unknown file>";
  const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
  auto KernelName = GetKernelName(Kernel);

  // Try to demangle the kernel name
  KernelName = DemangleName(KernelName);

  UR_LOG_L(getContext()->logger, QUIET,
           "====WARNING: DeviceSanitizer: data race");
  UR_LOG_L(getContext()->logger, QUIET,
           "When {} of size {} at {} in kernel <{}> LID({}, {}, {}) GID({}, "
           "{}, {})",
           Report.Type & kAccessRead ? "read" : "write", Report.AccessSize,
           (void *)Report.Address, KernelName.c_str(), Report.LID0, Report.LID1,
           Report.LID2, Report.GID0, Report.GID1, Report.GID2);
  UR_LOG_L(getContext()->logger, QUIET, "  #0 {} {}:{}", Func, File,
           Report.Line);
}

} // namespace tsan
} // namespace ur_sanitizer_layer
