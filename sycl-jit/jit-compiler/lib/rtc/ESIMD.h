//===-------------- ESIMD.h - Driver for ESIMD lowering -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_JIT_COMPILER_RTC_ESIMD_H
#define SYCL_JIT_COMPILER_RTC_ESIMD_H

#include "llvm/SYCLLowerIR/ModuleSplitter.h"

namespace jit_compiler {

// Runs a pass pipeline to lower ESIMD constructs on the given split model,
// which may only contain ESIMD entrypoints. This is a copy of the similar
// function in `sycl-post-link`.
void lowerEsimdConstructs(llvm::module_split::ModuleDesc &MD, bool PerformOpts);

} // namespace jit_compiler

#endif // SYCL_JIT_COMPILER_RTC_ESIMD_H
