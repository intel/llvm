//===--- SYCLStripDeadDebugInfo.h - Strip debug info from split module ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass goes through split module's metadata nodes list to detect those
// nodes that relate to dropped code and can be safely removed to reduce debug
// information size.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLStripDeadDebugInfo : public PassInfoMixin<SYCLStripDeadDebugInfo> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm
