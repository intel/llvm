//===-- CheckNDRangeSYCLNativeCPU.h  -Check if a kernel uses nd_range
//features--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Checks if the kernel uses features from nd_item such as:
// * local id
// * local range
// * local memory
// * work group barrier
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
