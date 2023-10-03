//===- RenameKernelSYCLNativeCPU.cpp - Kernel renaming for SYCL Native CPU-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass that renames the kernel names, to ensure the name
// doesn't clash with other names.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/RenameKernelSYCLNativeCPU.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include <set>

using namespace llvm;

static bool isSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.startswith("__spirv_") || FName.startswith("__sycl_");
}

PreservedAnalyses
RenameKernelSYCLNativeCPUPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  // Build the set of functions with sycl-module-id attr and functions
  // called by them
  std::set<Function *> CalledSet;
  SmallVector<Function *> workList;
  for (auto &F : M) {
    if (F.hasFnAttribute(sycl::utils::ATTR_SYCL_MODULE_ID)) {
      workList.push_back(&F);
    }
  }
  while (!workList.empty()) {
    auto *F = workList.pop_back_val();
    // skip SPIRV builtins and LLVM intrinsics
    if (isSpirvSyclBuiltin(F->getName()) || F->isIntrinsic())
      continue;
    auto Inserted = CalledSet.insert(F);
    if (!Inserted.second)
      continue;

    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *CBI = dyn_cast<CallBase>(&I)) {
          auto *Called = CBI->getCalledOperand();
          if (auto *CalledF = dyn_cast<Function>(Called)) {
            workList.push_back(CalledF);
          }
        }
      }
    }
  }

  for (auto &F : CalledSet) {
    F->setName(sycl::utils::addSYCLNativeCPUSuffix(F->getName()));
    ModuleChanged |= true;
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
