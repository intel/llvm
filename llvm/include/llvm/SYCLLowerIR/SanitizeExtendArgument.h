//===-- SanitizeExtendArgument.h - Append "__asan_launch" for sanitizer ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass appends a new argument "__asan_launch" to user's spir_kernel &
// spir_func, which is used to pass per-launch info for sanitized kernel.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"

namespace llvm {

class SanitizeExtendArgumentPass
    : public PassInfoMixin<SanitizeExtendArgumentPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm
