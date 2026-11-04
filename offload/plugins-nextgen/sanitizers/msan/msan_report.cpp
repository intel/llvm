//===--- msan_report.cpp - MSan error reports -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "msan_report.hpp"
#include "msan_libdevice.hpp"
#include "msan_origin.hpp"
#include "sanitizer_common.hpp"
#include "sanitizer_log.hpp"

namespace llvm::offload::sanitizer {
namespace msan {

bool ReportUsesUninitializedValue(const MsanErrorReport &Report,
                                  const std::string &KernelName) {
  const char *File = Report.File[0] ? Report.File : "<unknown file>";
  const char *Func = Report.Func[0] ? Report.Func : "<unknown func>";
  // Try to demangle the kernel name
  auto DemangledName = DemangleName(KernelName);

  SANITIZER_REPORT("====WARNING: DeviceSanitizer: use-of-uninitialized-value");

  SANITIZER_REPORT("use of size {} at kernel <{}> LID({}, {}, {}) GID({}, "
                   "{}, {})",
                   Report.AccessSize, DemangledName, Report.LID0, Report.LID1,
                   Report.LID2, Report.GID0, Report.GID1, Report.GID2);
  SANITIZER_REPORT("  #0 {} {}:{}", Func, File, Report.Line);

  if (!Report.Origin) {
    return true;
  }

  Origin Origin = Origin::FromRawId(Report.Origin);
  if (Origin.isHeapOrigin()) {
    HeapType Type = Origin.getHeapType();
    StackTrace Stack = Origin.getHeapStackTrace();
    SANITIZER_REPORT("ORIGIN: {} allocation ({})", ToString(Type),
                     (void *)(uptr)Report.Origin);
    Stack.print();

    return !(Type == HeapType::HostUSM || Type == HeapType::SharedUSM);
  }

  return true;
}

} // namespace msan
} // namespace llvm::offload::sanitizer
