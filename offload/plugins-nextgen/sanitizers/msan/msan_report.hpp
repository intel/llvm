//===--- msan_report.hpp - MSan error reports -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

namespace llvm::offload::sanitizer {

struct MsanErrorReport;

namespace msan {

// Abort the program if the return value is true
bool ReportUsesUninitializedValue(const MsanErrorReport &Report,
                                  const std::string &KernelName);

} // namespace msan
} // namespace llvm::offload::sanitizer
