//===-- CheckNDRangeSYCLNativeCPU.h  -Check if a kernel uses nd_range features--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass that:
// * Handles the kernel calling convention and attributes.
// * Materializes the spirv builtins so that they can be handled by the host
//   runtime.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

class CheckNDRangeSYCLNativeCPUPass
    : public PassInfoMixin<CheckNDRangeSYCLNativeCPUPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm
