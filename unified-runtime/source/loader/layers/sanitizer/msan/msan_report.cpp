/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_report.cpp
 *
 */

#include "msan_report.hpp"
#include "msan_libdevice.hpp"
#include "msan_origin.hpp"

#include "sanitizer_common/sanitizer_common.hpp"
#include "sanitizer_common/sanitizer_utils.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace msan {

bool ReportUsesUninitializedValue(const MsanErrorReport &Report,
                                  ur_kernel_handle_t Kernel) {
  const char *File = Report.File[0] ? Report.File : "<unknown file>";
  const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
  auto KernelName = GetKernelName(Kernel);

  // Try to demangle the kernel name
  KernelName = DemangleName(KernelName);

  UR_LOG_L(getContext()->logger, QUIET,
           "====WARNING: DeviceSanitizer: use-of-uninitialized-value");

  UR_LOG_L(getContext()->logger, QUIET,
           "use of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
           "{}, {})",
           Report.AccessSize, KernelName.c_str(), Report.LID0, Report.LID1,
           Report.LID2, Report.GID0, Report.GID1, Report.GID2);
  UR_LOG_L(getContext()->logger, QUIET, "  #0 {} {}:{}", Func, File,
           Report.Line);

  if (!Report.Origin) {
    return true;
  }

  Origin Origin = Origin::FromRawId(Report.Origin);
  if (Origin.isHeapOrigin()) {
    HeapType Type = Origin.getHeapType();
    StackTrace Stack = Origin.getHeapStackTrace();
    UR_LOG_L(getContext()->logger, QUIET, "ORIGIN: {} allocation ({})",
             ToString(Type), (void *)(uptr)Report.Origin);
    Stack.print();

    return !(Type == HeapType::HostUSM || Type == HeapType::SharedUSM);
  }

  return true;
}

} // namespace msan
} // namespace ur_sanitizer_layer
