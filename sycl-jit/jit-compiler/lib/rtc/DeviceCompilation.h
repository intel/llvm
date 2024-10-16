//==---- DeviceCompilation.h - Compile SYCL device code with libtooling ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_JIT_COMPILER_RTC_DEVICE_COMPILATION_H
#define SYCL_JIT_COMPILER_RTC_DEVICE_COMPILATION_H

#include "Kernel.h"
#include "View.h"

#include <llvm/Support/Error.h>

#include <memory>

namespace llvm {
class Module;
} // namespace llvm

namespace jit_compiler {

llvm::Expected<std::unique_ptr<llvm::Module>>
compileDeviceCode(const char *SYCLSource, View<IncludePair> IncludePairs,
                  View<const char *> UserArgs);

} // namespace jit_compiler

#endif // SYCL_JIT_COMPILER_RTC_DEVICE_COMPILATION_H
