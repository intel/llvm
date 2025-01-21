//===---- SYCLAddOptLevelAttribute.cpp - SYCLAddOptLevelAttribute Pass ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Pass adds 'sycl-optlevel' function attribute based on optimization level
// passed in.
//===---------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLAddOptLevelAttribute.h"

#include "llvm/IR/Module.h"

using namespace llvm;

PreservedAnalyses
SYCLAddOptLevelAttributePass::run(Module &M, ModuleAnalysisManager &MAM) {
  // Here, we add a function attribute 'sycl-optlevel' to store the
  // optimization level.
  assert(OptLevel >= 0 && "Invalid optimization level!");
  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;
    F.addFnAttr("sycl-optlevel", std::to_string(OptLevel));
  }
  return PreservedAnalyses::all();
}
