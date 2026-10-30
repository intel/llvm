//===--- sanitizer_log.hpp - Device sanitizer logging ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Shared/Debug.h"

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#define SANITIZER_REPORT(...)                                                  \
  (llvm::errs() << llvm::formatv(__VA_ARGS__) << '\n')

#define SANITIZER_LOG(...) ODBG("Sanitizer") << llvm::formatv(__VA_ARGS__)
