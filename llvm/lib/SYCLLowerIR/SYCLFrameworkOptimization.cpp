//=== SYCLFrameworkOptimization.cpp - Utility pass for framework optimization //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLFrameworkOptimization.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace sycl;

PreservedAnalyses AddDebugFuncAttrs::run(Module &M, ModuleAnalysisManager &) {
  for (Function &F : M) {
    if (!F.hasFnAttribute(Attribute::MinSize) && !F.hasFnAttribute(Attribute::AlwaysInline)) {
      F.addFnAttr(Attribute::NoInline);
      F.addFnAttr(Attribute::OptimizeNone);
    }
  }

  return PreservedAnalyses::all();
}
