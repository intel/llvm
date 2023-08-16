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

using namespace llvm;

PreservedAnalyses
RenameKernelSYCLNativeCPUPass::run(Module &M, ModuleAnalysisManager &MAM) {
  bool ModuleChanged = false;
  for (auto &F : M) {
    if (F.hasFnAttribute(sycl::utils::ATTR_SYCL_MODULE_ID)) {
      F.setName(sycl::utils::addSYCLNativeCPUSuffix(F.getName()));
    }
  }
  return ModuleChanged ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
