//===-- RenameKernelSYCLNativeCPU.h - Kernel renaming for SYCL Native CPU--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass that renames the kernel names, making sure that the
// mangled name is a string with no particular semantics.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

class RenameKernelSYCLNativeCPUPass
    : public PassInfoMixin<RenameKernelSYCLNativeCPUPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm
