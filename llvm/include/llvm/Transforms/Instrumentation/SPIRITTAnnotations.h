//===----- SPIRITTAnnotations.h - SPIR Instrumental Annotations Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which adds instrumental calls to annotate SPIR
// synchronization instructions. This can be used for kernel profiling.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class ModulePass;

class SPIRITTAnnotationsPass : public PassInfoMixin<SPIRITTAnnotationsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createSPIRITTAnnotationsLegacyPass();

} // namespace llvm
