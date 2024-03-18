//===-- SanitizeDeviceGlobal.h - instrument device global for sanitizer ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass adds red zone to each image scope device global and record the
// information like size, red zone size and beginning address. The information
// will be used by address sanitizer.
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"

namespace llvm {

class SanitizeDeviceGlobalPass
    : public PassInfoMixin<SanitizeDeviceGlobalPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm
